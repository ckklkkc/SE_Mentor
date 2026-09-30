import re
from typing import List, Dict

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from surya.recognition.model.config import SuryaOCRConfig

from langchain_opentutorial import set_env
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import OPENAI_API_KEY, LANGCHAIN_API_KEY
from common import FILEPATH

set_env(
    {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "LANGCHAIN_API_KEY": LANGCHAIN_API_KEY,
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "markdown-to-recursive",
    }
)

def pdf2md(file_path, chapter=None):
    # SuryaOCRConfig requires encoder/decoder arguments. Transformers 4.57
    # serializes it by first trying a no-argument default instance unless this
    # flag is set; that raises KeyError('encoder') while loading the OCR model.
    SuryaOCRConfig.has_no_defaults_at_init = True
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(file_path)
    markdown_output = rendered.markdown
    return markdown_output
    # output_name = f"md_files\\{chapter}_markdown.md"
    # with open(output_name, "w", encoding="utf-8") as f:
    #     f.write(markdown_output)

def md_splitter(md_content: str, with_recursive: bool = False):
    headers_to_split_on = [  
        ("#", "Header 1"),  
        ("##", "Header 2"),  
        ("###", "Header 3"),  
        ("####", "Header 4"),
    ]

    # Create a MarkdownHeaderTextSplitter object to split text based on markdown headers
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # Split markdown_document by headers and store in md_header_splits
    md_header_splits = markdown_splitter.split_text(md_content)

    if with_recursive:
        recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_chunks = recursive_splitter.split_documents(md_header_splits)
        return final_chunks
    
    return md_header_splits

def clean_markdown(text):
    # 移除圖片語法: ![替代文字](圖片連結)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # 移除連結語法，保留文字: [連結文字](連結) -> text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # 移除上標/下標或特殊 HTML 標籤
    text = re.sub(r'<.*?>', '', text)
    # 移除過多的空格與換行
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

def remove_specific_sections(text):
    re_list = [
        r'(^##\s*操作概念.*?)(?=^##\s|^#\s|\Z)', 
        r'(^##\s*使用者介面分析.*?)(?=^##\s|^#\s|\Z)', 
        r'(^##\s*版次變更記錄.*?)(?=^##\s|^#\s|\Z)', 
        r'(^##\s*目錄.*?)(?=^##\s|^#\s|\Z)',
        r'(^##\s*\d\.\s測試工作指派與時程.*?)(?=^##\s|^#\s|\Z)',
        r'(^##\s*\d\.\s測試環境.*?)(?=^##\s|^#\s|\Z)',
        r'(^##\s*\d\.\s測試目的與接受準則.*?)(?=^##\s|^#\s|\Z)'
    ]

    for r in re_list:
        text = re.sub(r, '', text, flags=re.MULTILINE | re.DOTALL)

    return text

def detect_and_extract_tables(text: str) -> List[str]:
    lines = text.split('\n')
    tables = []
    current_table = []
    in_table = False
    
    for line in lines:
        # 檢查是否是表格行（以 | 開頭和結尾）
        if line.strip().startswith('|') and line.strip().endswith('|'):
            # 檢查是否是分隔線
            if re.match(r'^\|\s*[-:\s|]+\s*\|$', line):
                # 這是分隔線
                if current_table:
                    current_table.append(line)
                    in_table = True
            else:
                # 這是數據行或表頭
                if not in_table and current_table:
                    # 已經完成一個表格，開始新的
                    tables.append('\n'.join(current_table))
                    current_table = [line]
                    in_table = False
                else:
                    current_table.append(line)
        else:
            # 非表格行
            if current_table and in_table:
                # 表格已結束
                tables.append('\n'.join(current_table))
                current_table = []
                in_table = False
    
    # 最後的表格
    if current_table:
        tables.append('\n'.join(current_table))
    
    return tables

def parse_markdown_table(table: str) -> List[Dict[str, str]]:
    """
    將 Markdown 表格解析為字典列表
    
    Args:
        markdown_text: 包含 Markdown 表格的文本
        
    Returns:
        列表，每個元素是一個字典，代表表格的一行
    """
    lines = table.strip().split('\n')
    
    if len(lines) < 3:
        raise ValueError("無效的 Markdown 表格格式")
    
    # 解析表頭
    header_line = lines[0]
    headers = [h.strip() for h in header_line.split('|')[1:-1]]  # 移除開頭和結尾的空字符串
    
    # 解析表格資料內容
    data_rows = []
    for line in lines[2:]:
        if line.strip():  # 跳過空行
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            # 如果列數與表頭不符，進行調整
            if len(cells) != len(headers):
                print(f"警告：行的列數 ({len(cells)}) 與表頭列數 ({len(headers)}) 不符")
                print(line)
            
            # 建立字典，缺失的欄位設為空字符串
            row_dict = {}
            for i, header in enumerate(headers):
                row_dict[header] = cells[i] if i < len(cells) else ""
            data_rows.append(row_dict)
    
    return headers, data_rows


def format_table_as_form(headers: List[str], data_rows: List[Dict[str, str]]) -> str:
    """
    將解析的表格數據格式化為表單形式
    
    Args:
        headers: 表頭列表
        data_rows: 數據行列表
        format_type: 格式類型 ('simple' 或 'markdown')
        
    Returns:
        格式化後的文本
    """
    result = []
    
    for idx, row in enumerate(data_rows, 1):
        result.append("")  # 添加空行以分隔不同的記錄
        
        for header in headers:
            value = row.get(header, "")
            result.append(f"{header}：{value}")
        
        result.append("---")
    
    return "\n".join(result)
 
def replace_tables_in_text(text: str) -> str:
    """
    找到文本中的所有表格，將其替換為表單形式
    
    Args:
        text: 輸入文本
        
    Returns:
        替換後的文本
    """
    lines = text.split('\n')
    result_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # 檢查是否是表格行
        if line.strip().startswith('|') and line.strip().endswith('|'):
            # 找到表格的開始
            table_lines = [line]
            i += 1
            
            # 檢查下一行是否是分隔線
            if i < len(lines) and re.match(r'^\|\s*[-:\s|]+\s*\|$', lines[i]):
                table_lines.append(lines[i])
                i += 1
                
                # 收集表格的所有數據行
                while i < len(lines) and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                    table_lines.append(lines[i])
                    i += 1
                
                # 現在有完整的表格，轉換它
                try:
                    table_text = '\n'.join(table_lines)
                    headers, data_rows = parse_markdown_table(table_text)
                    # for row in data_rows:
                    #     print(row)
                    # print("="*30)
                    form_text = format_table_as_form(headers, data_rows)
                    result_lines.append(form_text)
                except Exception as e:
                    print(f"警告：無法解析表格，保留原始內容。錯誤：{e}")
                    result_lines.extend(table_lines)
            else:
                # 不是有效的表格，保留原始內容
                result_lines.extend(table_lines)
        else:
            # 非表格行，直接添加
            result_lines.append(line)
            i += 1
    
    return '\n'.join(result_lines)

def replace_tables_in_text_v2(text: str) -> str:
    """
    替代版本 - 使用正則替換（更簡潔）
    """
    def replace_match(match):
        table_text = match.group(0)
        try:
            headers, data_rows = parse_markdown_table(table_text)
            return format_table_as_form(headers, data_rows)
        except Exception as e:
            print(f"警告：無法解析表格：{e}")
            return table_text
    
    # 匹配完整的表格
    pattern = r'(\|.+\|\s*\n\|[\s\-|:]+\|\s*\n(?:\|.+\|\s*\n?)*)'
    return re.sub(pattern, replace_match, text)

class Tags(BaseModel):
    tags: list[str] = Field(description="與文本有關的標籤列表")

def get_tags_from_gpt(course_name, level, doc_content):
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    parser = JsonOutputParser(pydantic_object=Tags)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位專業的{course_name}課程助教，請為以下內容提取3個{level}中文關鍵字標籤，並將3個標籤轉為英文一起輸出，如果原本的標籤是英文，則不須翻譯。"),
            ("user", "#Format: {format_instructions}\n\n#Content: {content}"),
        ]
    )

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser

    answer = chain.invoke({"course_name": course_name, "level": level, "content": doc_content})

    return answer

if __name__ == '__main__':
    # file_names = ["[08]軟體設計-模組設計", "[09]軟體測試", "[10]進階軟體測試", "[11]DevOps自動化建置管理"]
    # chapters = [8, 9, 10, 11]
    # for file_name, ch in zip(file_names, chapters):
    #     pdf2md(f"C:\\Users\\shanyiii\\Desktop\\mine\\1141軟體工程\\slides\\{file_name}.pdf", f"ch{ch}")

    try:
        with open(f"md_files\\document\\海大餐飲外送系統-軟體設計文件(SDD)-table.md", 'r', encoding='utf-8') as input_file:
            md_content = input_file.read()
    except FileNotFoundError:
        print("Error: The specified file was not found.")

    cleaned_md = clean_markdown(md_content)
    md_documents = md_splitter(cleaned_md)

    for doc in md_documents:
        # print(doc.page_content)
        print("="*30)
        table_string = replace_tables_in_text(doc.page_content)
        print(table_string)
