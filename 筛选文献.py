import streamlit as st
import os
import pypdf
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# 1. 基础配置与工具函数
# ==========================================
st.set_page_config(page_title="药学文献智能筛选平台", layout="wide", page_icon="💊")

def extract_text_from_pdf(uploaded_file):
    """辅助函数：将上传的PDF文件转换为字符串文本"""
    try:
        pdf_reader = pypdf.PdfReader(uploaded_file)
        text = ""
        # 为了防止Token溢出，这里可以限制读取前N页，或者读取全部
        # 这里默认读取全部，DeepSeek窗口很大，通常能这就hold住
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

# ==========================================
# 2. 定义 Agent 状态 (State)
# ==========================================
class LiteratureState(TypedDict):
    file_name: str          # 文件名
    raw_content: str        # PDF 提取出的原文
    screening_criteria: str # 用户设定的筛选标准（变量）
    extracted_data: str     # 筛选出的数据
    quality_report: str     # 监控者的评分报告

# ==========================================
# 3. 核心处理逻辑 (封装供 Streamlit 调用)
# ==========================================
def process_document(api_key, model_name, file_obj, criteria):
    
    # 设置环境变量
    os.environ["DEEPSEEK_API_KEY"] = api_key
    
    # 初始化模型
    # 注意：DeepSeek 的推理模型通常叫 deepseek-reasoner，通用模型叫 deepseek-chat
    llm = ChatOpenAI(
        model=model_name, 
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0  # 科研数据要求严谨
    )

    # --- 节点 1: 抓取与预处理智能体 ---
    # 负责将 PDF 对象转化为 LLM 可读的文本
    def pdf_loader_agent(state: LiteratureState):
        # 这一步其实在传入前已经由工具函数辅助完成了，
        # 但在逻辑上，我们可以在这里做进一步的数据清洗（如去掉页眉页脚）
        text = state["raw_content"]
        # 简单清洗：去掉过多的空行
        clean_text = "\n".join([line for line in text.split('\n') if line.strip()])
        return {"raw_content": clean_text}

    # --- 节点 2: 筛选分析智能体 ---
    # 根据用户动态设定的 criteria 进行提取
    def filter_agent(state: LiteratureState):
        prompt = f"""
        你是一名专业的药学数据分析师。
        任务：请根据以下【筛选标准】，从【文献内容】中提取精确的数据。
        
        【筛选标准】: 
        {state["screening_criteria"]}
        
        【文献内容】(部分展示):
        {state["raw_content"][:30000]} ... (内容过长已截断，请基于全量理解)
        
        要求：
        1. 只输出提取到的数据结果，可以是表格形式或列表形式。
        2. 如果文中未提及某项标准，请明确标注“未找到”。
        3. 不要输出无关的寒暄语。
        """
        # 注意：实际发送时建议发送完整 content，这里为了演示 Prompt 结构
        # 真实调用使用完整文本
        real_msg = prompt.replace(f"{state['raw_content'][:30000]} ... (内容过长已截断，请基于全量理解)", state["raw_content"])
        
        response = llm.invoke([HumanMessage(content=real_msg)])
        return {"extracted_data": response.content}

    # --- 节点 3: 监督监控智能体 (Thinker) ---
    # 负责检查准确率，DeepSeek-R1 (Reasoner) 在此类反思任务上表现优异
    def monitor_agent(state: LiteratureState):
        prompt = f"""
        你是一名严格的科研质量监督员。
        
        你的任务是审核上一步的【提取结果】是否忠实于【文献原文】以及是否符合【筛选标准】。
        
        【用户标准】: {state["screening_criteria"]}
        【提取结果】: {state["extracted_data"]}
        【文献原文片段】: {state["raw_content"][:5000]}...
        
        请输出一份简短的【质量监控报告】：
        1. 准确率评分 (0-100)。
        2. 是否存在幻觉或遗漏？
        3. 最终修正建议（如有）。
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"quality_report": response.content}

    # --- 构建图 ---
    workflow = StateGraph(LiteratureState)
    
    workflow.add_node("PDF_Loader", pdf_loader_agent)
    workflow.add_node("Filter", filter_agent)
    workflow.add_node("Monitor", monitor_agent)
    
    workflow.add_edge(START, "PDF_Loader")
    workflow.add_edge("PDF_Loader", "Filter")
    workflow.add_edge("Filter", "Monitor")
    workflow.add_edge("Monitor", END)
    
    app = workflow.compile()
    
    # 预先读取 PDF 文本
    raw_text = extract_text_from_pdf(file_obj)
    
    # 启动工作流
    inputs = {
        "file_name": file_obj.name,
        "raw_content": raw_text,
        "screening_criteria": criteria,
        "extracted_data": "",
        "quality_report": ""
    }
    
    return app.invoke(inputs)

# ==========================================
# 4. Streamlit 界面构建
# ==========================================

# 侧边栏
with st.sidebar:
    st.header("⚙️ 全局设置")
    api_key = st.text_input("DeepSeek API Key", type="password")
    
    # 让用户选择模型：如果你的账号支持推理模型，选 reasoner 效果更好
    model_choice = st.selectbox(
        "选择模型能力", 
        ("deepseek-chat (快速)", "deepseek-reasoner (深度思考)")
    )
    # 映射到真实的 API model name
    model_map = {
        "deepseek-chat (快速)": "deepseek-chat",
        "deepseek-reasoner (深度思考)": "deepseek-reasoner"
    }
    selected_model = model_map[model_choice]

    st.markdown("---")
    st.info("💡 **提示**: \nDeepSeek-V3 (chat) 适合快速提取。\nDeepSeek-R1 (reasoner) 适合复杂的逻辑校验。")

st.title("💊 药学文献批量智能筛选系统")
st.markdown("---")

# 1. 变量设置区 (用户需求的核心)
st.subheader("1. 设定筛选标准 (变量定义)")
default_criteria = """
请提取以下信息：
1. 药物名称 (Drug Name)
2. 实验组样本量 (Sample Size)
3. 核心不良反应 (Adverse Events)
4. P值 (P-value)
"""
criteria_input = st.text_area("在此定义你想从文献中挖掘什么数据：", value=default_criteria, height=150)

# 2. 文件上传区
st.subheader("2. 上传文献 (支持批量)")
uploaded_files = st.file_uploader("请上传 PDF 文件", type=["pdf"], accept_multiple_files=True)

# 3. 执行按钮
if st.button("🚀 开始批量分析", type="primary"):
    if not api_key:
        st.error("请先在左侧侧边栏输入 API Key！")
    elif not uploaded_files:
        st.warning("请至少上传一个 PDF 文件。")
    else:
        # 创建一个进度条
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        
        st.markdown("### 📊 分析结果看板")
        
        # 循环处理每个文件
        for idx, pdf_file in enumerate(uploaded_files):
            with st.expander(f"📄 文件: {pdf_file.name}", expanded=True):
                with st.spinner(f"正在读取并分析 {pdf_file.name} ..."):
                    try:
                        # 调用 Agent 系统
                        result = process_document(api_key, selected_model, pdf_file, criteria_input)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("🔍 数据提取结果")
                            st.markdown(result["extracted_data"])
                            
                        with col2:
                            st.subheader("🛡️ 监督者报告")
                            #以此不同颜色显示，增强警示作用
                            st.info(result["quality_report"])
                            
                    except Exception as e:
                        st.error(f"处理失败: {e}")
            
            # 更新进度条
            progress_bar.progress((idx + 1) / total_files)
            
        st.success("✅ 所有文献处理完毕！")    