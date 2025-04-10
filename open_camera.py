from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
import os
import time
import pyautogui
import re
import subprocess
from datetime import datetime

@tool
def open_camera(query: str) -> str:
    """Tool to open the Windows Camera application. Use this tool when the user wants to open, launch, 
    start or use the camera. (工具用於打開Windows相機應用程式。當用戶想要打開、啟動或使用相機時使用此工具。)"""
    try:
        os.system("start microsoft.windows.camera:")
        # 給相機應用足夠時間打開
        time.sleep(2)
        return "The camera application has been opened. (已打開相機應用程式)"
    except Exception as e:
        return f"An error occurred while opening the camera: {e} (打開相機時發生錯誤：{e})"

@tool
def take_photo(save_path: str = "") -> str:
    """Tool to take a photo using the Windows Camera application and save it to a specified path.
    If no path is specified, it will save to the default Pictures folder.
    (工具用於使用Windows相機應用程式拍照並保存到指定路徑。如果未指定路徑，將保存到默認圖片文件夾。)"""
    try:
        # 確保相機應用已打開
        if not is_camera_open():
            open_camera("")
            time.sleep(2)
        
        # 使用Windows PowerShell的SendKeys來模擬空格鍵按下（拍照）
        powershell_command = 'powershell -command "$wshell = New-Object -ComObject wscript.shell; $wshell.AppActivate(\'Camera\'); Start-Sleep -m 500; $wshell.SendKeys(\' \');"'
        subprocess.run(powershell_command, shell=True)
        time.sleep(1)  # 等待照片被拍攝
        
        # 如果指定了保存路徑
        if save_path:
            # 創建目標目錄（如果不存在）
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # 獲取最新拍攝的照片
            default_path = os.path.expanduser("~\\Pictures\\Camera Roll")
            if os.path.exists(default_path):
                files = [os.path.join(default_path, f) for f in os.listdir(default_path) if f.endswith('.jpg') or f.endswith('.png')]
                if files:
                    newest_file = max(files, key=os.path.getctime)
                    # 複製到目標位置
                    import shutil
                    shutil.copy2(newest_file, save_path)
                    return f"Photo taken and saved to {save_path} (已拍照並保存到 {save_path})"
            
            return f"Photo taken, but couldn't save to specified path. Check default camera folder. (已拍照，但無法保存到指定路徑。請檢查默認相機文件夾。)"
        
        return "Photo taken and saved to default location. (已拍照並保存到默認位置。)"
    except Exception as e:
        return f"An error occurred while taking a photo: {e} (拍照時發生錯誤：{e})"

@tool
def wait_seconds(seconds: str) -> str:
    """Tool to wait for a specified number of seconds.
    (工具用於等待指定的秒數。)"""
    try:
        # 從輸入中提取數字
        match = re.search(r'(\d+)', seconds)
        if match:
            wait_time = int(match.group(1))
        else:
            wait_time = 5  # 默認等待5秒
        
        time.sleep(wait_time)
        return f"Waited for {wait_time} seconds. (已等待 {wait_time} 秒。)"
    except Exception as e:
        return f"An error occurred while waiting: {e} (等待時發生錯誤：{e})"

@tool
def close_camera(query: str) -> str:
    """Tool to close the Windows Camera application. Use this tool when the user wants to close or exit the camera.
    (工具用於關閉Windows相機應用程式。當用戶想要關閉或退出相機時使用此工具。)"""
    try:
        os.system("taskkill /f /im WindowsCamera.exe")
        return "Camera application has been closed. (相機應用程式已關閉。)"
    except Exception as e:
        return f"An error occurred while closing the camera: {e} (關閉相機時發生錯誤：{e})"

def is_camera_open() -> bool:
    """Check if the Windows Camera application is currently open.
    (檢查Windows相機應用程式是否當前打開。)"""
    output = subprocess.check_output('tasklist', shell=True).decode()
    return 'WindowsCamera.exe' in output

@tool
def execute_camera_sequence(command: str) -> str:
    """Tool to execute a sequence of camera operations: open camera, wait, take photo, save to location.
    Example: "開啟相機 等待5秒 拍一張照片 並存到C:\\Users\\username\\Documents\\photo.png"
    (工具用於執行一系列相機操作：打開相機，等待，拍照，保存到位置。)"""
    try:
        # 檢查命令是否包含打開相機的指令
        if any(keyword in command for keyword in ["開啟相機", "打開相機", "啟動相機", "open camera", "start camera"]):
            result = open_camera("")
            print(result)
        
        # 檢查等待時間
        wait_match = re.search(r'等待(\d+)秒|wait (\d+) seconds', command, re.IGNORECASE)
        if wait_match:
            seconds = wait_match.group(1) or wait_match.group(2)
            result = wait_seconds(seconds)
            print(result)
        
        # 檢查是否要拍照
        if any(keyword in command for keyword in ["拍一張照片", "拍照", "take photo", "capture"]):
            # 檢查保存路徑
            save_match = re.search(r'存到(.*\.png|.*\.jpg)|save to (.*\.png|.*\.jpg)', command)
            if save_match:
                save_path = save_match.group(1) or save_match.group(2)
                save_path = save_path.strip()
                result = take_photo(save_path)
            else:
                result = take_photo("")
            print(result)
            
        return "Camera sequence executed successfully. (相機操作序列已成功執行。)"
    except Exception as e:
        return f"An error occurred during camera sequence: {e} (相機操作序列期間發生錯誤：{e})"

def main():
    # 使用llama3或其他具有更好多語言能力的模型
    llm = OllamaLLM(model="mistral", temperature=0)  # 較低的temperature以獲得更確定性的響應
    
    tools = [open_camera, take_photo, wait_seconds, close_camera, execute_camera_sequence]
    
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
    )
    
    print("AI Camera Assistant (AI 相機助手)")
    print("-----------------------------------------")
    print("Type 'exit' or '退出' to quit (輸入 'exit' 或 '退出' 來結束)")
    print("Example: '開啟相機 等待5秒 拍一張照片 並存到C:\\Users\\username\\Documents\\photo.png'")
    
    while True:
        user_input = input("\nEnter your prompt (輸入您的命令): ")
        
        if user_input.lower() in ["exit", "退出"]:
            print("Exiting the program. Goodbye! (退出程序。再見！)")
            break
        
        try:
            # 檢查是否是直接的相機操作序列
            if any(keyword in user_input for keyword in ["開啟相機", "等待", "拍一張照片", "存到"]):
                result = execute_camera_sequence(user_input)
                print("\nResult:", result)
            else:
                result = agent.run(user_input)
                print("\nResult:", result)
        except Exception as e:
            print(f"\nError: {e}")
            print("Opening camera directly as fallback...")
            result = open_camera("fallback")
            print("\nResult:", result)

if __name__ == "__main__":
    main()