from google import genai
import os
from dotenv import load_dotenv
import time

# 1. .env 파일 로드
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

# 2. Gemini 클라이언트 초기화
client = genai.Client(api_key=API_KEY)

def extract_text_from_pdf_gemini(pdf_path: str) -> str:
    """Gemini로 PDF → 텍스트 변환"""
    # 파일 업로드
    uploaded_file = client.files.upload(file=pdf_path)
    
    # 텍스트 추출 요청
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "이 PDF의 모든 텍스트를 추출해주세요. 테이블은 마크다운으로 변환해주세요.",
            uploaded_file
        ]
    )
    
    # 업로드된 파일 삭제
    client.files.delete(name=uploaded_file.name)
    
    return response.text

def convert_folder_pdfs_to_txt(pdf_folder: str, txt_folder: str) -> None:
    """
    주어진 폴더의 모든 PDF를 변환해 결과를 txt 파일로 저장
    :param pdf_folder: PDF 파일들이 있는 폴더 경로
    :param txt_folder: 변환된 txt 파일을 저장할 폴더 경로
    """
    # 출력 폴더가 없으면 생성
    os.makedirs(txt_folder, exist_ok=True)

    # 폴더 내 모든 .pdf 파일을 순회
    for file_name in os.listdir(pdf_folder):
        if not file_name.lower().endswith(".pdf"):
            continue
        
        pdf_path = os.path.join(pdf_folder, file_name)
        base_name = os.path.splitext(file_name)[0]
        txt_path = os.path.join(txt_folder, f"{base_name}.txt")
        
        try:
            print(f"변환 중: {pdf_path}")
            text = extract_text_from_pdf_gemini(pdf_path)
            
            # 결과를 txt로 저장
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            print(f"저장 완료: {txt_path}")
            time.sleep(30)
        except Exception as e:
            print(f"오류 발생 ({pdf_path}): {e}")

if __name__ == "__main__":
    # 사용 예드
    PDF_FOLDER = "실행_메뉴얼"        # 변환할 PDF들이 들어있는 폴더
    TXT_FOLDER = "txt_outputs" # 변환된 txt를 저장할 폴더
    convert_folder_pdfs_to_txt(PDF_FOLDER, TXT_FOLDER)
