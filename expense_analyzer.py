from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
import re
from collections import defaultdict
import PyPDF2
import traceback

# Define tools for financial analysis
@tool
def analyze_bank_statement(file_path: str) -> str:
    """
    Analyzes a bank statement to generate expense categories, visualizations, and financial recommendations.
    Use this tool when the user asks to analyze a bank statement, financial document, or wants expense categorization with charts.
    Input should be a file path to a financial document (PDF, CSV, Excel).
    (分析銀行對帳單以生成支出分類、視覺化圖表和財務建議。當用戶要求分析銀行對帳單、財務文件或需要帶有圖表的支出分類時使用此工具。)
    """
    # Remove any quotes from the path that might be included
    file_path = file_path.strip('"\'')
    
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path} (錯誤：找不到文件)"
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Call the appropriate analysis function based on file type
    if file_ext == '.pdf':
        result = analyze_expenses(file_path)
    elif file_ext in ['.csv', '.xlsx', '.xls']:
        result = analyze_expenses(file_path)
    else:
        return f"Unsupported file format: {file_ext} (不支持的文件格式)"
    
    if result:
        df = result['dataframe']
        
        # Create a summary string to return
        summary = "===== 財務分析報告 / Financial Analysis Report =====\n\n"
        
        # Filter for expenses and income
        expenses_df = df[df['amount'] < 0].copy()
        expenses_df['amount'] = expenses_df['amount'].abs()  # Make positive for display
        income_df = df[df['amount'] > 0]
        
        total_expense = expenses_df['amount'].sum() if not expenses_df.empty else 0
        total_income = income_df['amount'].sum() if not income_df.empty else 0
        
        summary += f"總收入 / Total income: {total_income:,.2f}\n"
        summary += f"總支出 / Total expenses: {total_expense:,.2f}\n"
        summary += f"淨值 / Net: {total_income - total_expense:,.2f}\n\n"
        
        # Add expense categories
        if not expenses_df.empty:
            summary += "===== 支出類別 / Expense Categories =====\n"
            category_summary = expenses_df.groupby('category')['amount'].agg(['sum', 'count'])
            category_summary['percentage'] = category_summary['sum'] / total_expense * 100
            category_summary = category_summary.sort_values('sum', ascending=False)
            
            for category, row in category_summary.iterrows():
                summary += f"{category}: {row['sum']:,.2f} ({row['percentage']:.1f}% of total), {int(row['count'])} transactions\n"
            
            summary += "\n"
        
        # Add recommendations
        summary += "===== 財務建議 / Financial Recommendations =====\n"
        for rec in result['recommendations']:
            summary += f"- {rec}\n"
        
        summary += "\n圖表已保存為 'sinopac_expense_analysis.png' / Visualizations saved as 'sinopac_expense_analysis.png'\n"
        
        return summary
    else:
        return "無法從文件中提取資料進行分析 / Unable to extract data from the document for analysis"

@tool
def open_camera(query: str) -> str:
    """Open the Windows Camera application (打開 Windows 相機應用程式)"""
    try:
        os.system("start microsoft.windows.camera:")
        return "The camera application has been opened. (已打開相機應用程式)"
    except Exception as e:
        return f"An error occurred while opening the camera: {e} (打開相機時發生錯誤：{e})"

# Financial analysis functions
def extract_from_pdf(pdf_path):
    """Extracts transaction data from SinoPac Bank PDF statement."""
    print("Extracting data from SinoPac Bank PDF statement...")
    
    # Open PDF and extract text
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        
        # Parse the text into transactions
        transactions = []
        
        # Look for transaction patterns (date, description, amount)
        # Pattern for SinoPac Bank statement transaction rows
        pattern = r'(\d{4}/\d{2}/\d{2})\s+([\w\s\-－ＫＹ]+?)\s+(\d+,?\d*\.?\d*)\s+(\d+,?\d*\.?\d*)\s+(\d+,?\d*\.?\d*)'
        matches = re.findall(pattern, text)
        
        for match in matches:
            date, description, out_amount, in_amount, balance = match
            
            # Determine if this is an outgoing or incoming transaction
            amount = 0
            transaction_type = ""
            
            if out_amount.strip():
                amount = -float(out_amount.replace(',', ''))
                transaction_type = "outgoing"
            elif in_amount.strip():
                amount = float(in_amount.replace(',', ''))
                transaction_type = "incoming"
            
            # Only add valid transactions
            if amount != 0:
                transactions.append({
                    'date': date,
                    'description': description.strip(),
                    'amount': amount,
                    'balance': float(balance.replace(',', '')),
                    'type': transaction_type
                })
        
        if transactions:
            print(f"Extracted {len(transactions)} transactions from the PDF")
            return pd.DataFrame(transactions)
        else:
            print("Could not extract transactions from PDF text")
            return None
            
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None

def preprocess_expense_data(df):
    """Preprocesses and cleans the SinoPac Bank transaction data."""
    if df is None or df.empty:
        return pd.DataFrame(columns=['date', 'description', 'amount', 'category'])
    
    # Ensure columns are correctly formatted
    std_df = pd.DataFrame()
    
    # Process date column
    if 'date' in df.columns:
        std_df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d', errors='coerce')
    
    # Process description and amount columns
    if 'description' in df.columns:
        std_df['description'] = df['description']
    
    if 'amount' in df.columns:
        std_df['amount'] = df['amount']
    
    # Drop rows with missing key data
    std_df = std_df.dropna(subset=['amount'])
    
    return std_df

def categorize_expenses(df):
    """Categorizes expenses based on description text for SinoPac Bank transactions."""
    # Define category keywords - tailored for SinoPac Bank statements
    categories = {
        '股票/Stocks': ['股票', '元大', '證券', 'ETF', '基金', 'FUND', '譜瑞－ＫＹ', '皇昌', '智原', '新美齊', '和大',
                       '康舒', '致新', '威剛', '國巨', '南電', '華孚', '榮運', '凱銳光電'],
        '銀行費用/Bank Fees': ['手續費', '申購', '預扣', '利息', '退還', 'ACH', '折讓'],
        '信用卡/Credit Card': ['永豐卡費', '信用卡'],
        '生活支出/Daily Expenses': ['悠遊戶外運費'],
        '轉帳/Transfers': ['手機轉帳', '轉帳'],
        '股息/Dividends': ['股息', 'ACH股息', '科技優息'],
        '回饋/Rebates': ['大戶回饋'],
        '其他/Others': []  # Default category
    }
    
    # Function to determine category
    def determine_category(description):
        if not isinstance(description, str):
            return '其他/Others'
            
        description = str(description).lower()
        for category, keywords in categories.items():
            if any(keyword.lower() in description for keyword in keywords):
                return category
        return '其他/Others'
    
    # Apply categorization
    df['category'] = df['description'].apply(determine_category)
    
    return df

def create_expense_visualizations(df):
    """Creates visualizations for expense analysis."""
    plt.figure(figsize=(16, 10))
    
    # Set style with Chinese font support
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    
    # Filter for expenses (negative amounts)
    expenses_df = df[df['amount'] < 0].copy()
    expenses_df['amount'] = expenses_df['amount'].abs()  # Make positive for visualization
    
    # Filter for income (positive amounts)
    income_df = df[df['amount'] > 0].copy()
    
    # 1. Pie chart for expense categories
    plt.subplot(2, 2, 1)
    if not expenses_df.empty:
        category_totals = expenses_df.groupby('category')['amount'].sum().sort_values(ascending=False)
        plt.pie(category_totals, labels=category_totals.index, autopct='%1.1f%%', startangle=90)
        plt.title('支出類別分佈 / Expense Categories Distribution', fontsize=14)
        plt.axis('equal')
    else:
        plt.text(0.5, 0.5, '沒有支出數據 / No expense data', horizontalalignment='center', fontsize=14)
        plt.axis('off')
    
    # 2. Bar chart for top expenses by category
    plt.subplot(2, 2, 2)
    if not expenses_df.empty:
        category_totals = expenses_df.groupby('category')['amount'].sum().sort_values(ascending=False)
        category_totals.plot(kind='bar', color=sns.color_palette("viridis", len(category_totals)))
        plt.title('各類別支出金額 / Expense Amount by Category', fontsize=14)
        plt.xlabel('類別 / Category')
        plt.ylabel('金額 / Amount')
        plt.xticks(rotation=45, ha='right')
    else:
        plt.text(0.5, 0.5, '沒有支出數據 / No expense data', horizontalalignment='center', fontsize=14)
        plt.axis('off')
    
    # 3. Time series of daily expenses
    plt.subplot(2, 2, 3)
    if 'date' in df.columns and not df.empty:
        # Daily total expenses
        expenses_by_date = expenses_df.groupby(expenses_df['date'].dt.date)['amount'].sum()
        
        # Only plot if we have data
        if not expenses_by_date.empty:
            plt.bar(expenses_by_date.index, expenses_by_date.values, color='crimson')
            plt.title('每日支出 / Daily Expenses', fontsize=14)
            plt.xlabel('日期 / Date')
            plt.ylabel('金額 / Amount')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, '沒有日期支出數據 / No dated expense data', horizontalalignment='center', fontsize=14)
            plt.axis('off')
    else:
        plt.text(0.5, 0.5, '沒有日期數據 / No date data', horizontalalignment='center', fontsize=14)
        plt.axis('off')
    
    # 4. Income vs expense overview
    plt.subplot(2, 2, 4)
    if not df.empty:
        # Calculate totals
        total_income = income_df['amount'].sum() if not income_df.empty else 0
        total_expense = expenses_df['amount'].sum() if not expenses_df.empty else 0
        
        # Create bar chart
        plt.bar(['收入 / Income', '支出 / Expenses'], [total_income, total_expense], color=['green', 'red'])
        plt.title('收入與支出總覽 / Income vs Expenses Overview', fontsize=14)
        plt.ylabel('金額 / Amount')
        
        # Add net value as text
        net = total_income - total_expense
        plt.text(1.5, max(total_income, total_expense) * 0.5, 
                f'淨值 / Net: {net:,.2f}', 
                fontsize=12, ha='center')
    else:
        plt.text(0.5, 0.5, '沒有數據 / No data', horizontalalignment='center', fontsize=14)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sinopac_expense_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("分析圖表已創建並保存為 'sinopac_expense_analysis.png' / Visualizations created and saved as 'sinopac_expense_analysis.png'")

def generate_recommendations(df):
    """Generates tailored financial recommendations based on SinoPac Bank statement analysis."""
    recommendations = []
    
    if df.empty:
        recommendations.append("沒有足夠的交易數據進行分析。 / Not enough transaction data for analysis.")
        return recommendations
    
    # Filter for expenses (negative amounts)
    expenses_df = df[df['amount'] < 0].copy()
    expenses_df['amount'] = expenses_df['amount'].abs()  # Make positive for analysis
    
    # Filter for income (positive amounts)
    income_df = df[df['amount'] > 0].copy()
    
    # Get total expense and income
    total_expense = expenses_df['amount'].sum() if not expenses_df.empty else 0
    total_income = income_df['amount'].sum() if not income_df.empty else 0
    
    # Analyze stock trading patterns
    stock_transactions = df[df['category'] == '股票/Stocks']
    if len(stock_transactions) > 0:
        stock_expenses = stock_transactions[stock_transactions['amount'] < 0]['amount'].abs().sum()
        stock_income = stock_transactions[stock_transactions['amount'] > 0]['amount'].sum()
        
        # Calculate stock trading metrics
        if stock_expenses > 0:
            stock_turnover_ratio = (stock_expenses + stock_income) / stock_expenses
            
            if stock_turnover_ratio > 3:
                recommendations.append("您的股票交易頻率較高，考慮減少交易次數以降低交易成本。")
                recommendations.append("Your stock trading frequency is high. Consider reducing the number of trades to lower transaction costs.")
            
            stock_percent = stock_expenses / total_expense * 100 if total_expense > 0 else 0
            if stock_percent > 70:
                recommendations.append(f"股票交易支出佔總支出的{stock_percent:.1f}%，建議適度分散投資組合。")
                recommendations.append(f"Stock trading expenses account for {stock_percent:.1f}% of your total spending. Consider diversifying your investment portfolio.")
    
    # Analyze transfer patterns
    transfers = df[df['category'] == '轉帳/Transfers']
    if len(transfers) > 0:
        large_transfers = transfers[transfers['amount'].abs() > 10000]
        if len(large_transfers) > 2:
            recommendations.append("本月有多筆大額轉帳，建議確認這些資金流動的目的，並考慮建立更有系統的預算計劃。")
            recommendations.append("There are multiple large transfers this month. Verify the purpose of these fund movements and consider establishing a more systematic budget plan.")
    
    # General financial health recommendations
    if total_income > 0 and total_expense > 0:
        savings_rate = (total_income - total_expense) / total_income * 100
        
        if savings_rate < 10:
            recommendations.append(f"您的儲蓄率為{savings_rate:.1f}%，低於理想的20%水平。考慮增加儲蓄以備不時之需。")
            recommendations.append(f"Your savings rate is {savings_rate:.1f}%, below the ideal 20% level. Consider increasing your savings for emergencies.")
        elif savings_rate > 50:
            recommendations.append(f"您的儲蓄率為{savings_rate:.1f}%，相當高。可以考慮將部分資金用於長期投資。")
            recommendations.append(f"Your savings rate is {savings_rate:.1f}%, which is quite high. Consider allocating some funds to long-term investments.")
    
    # Credit card recommendations
    credit_card = df[df['category'] == '信用卡/Credit Card']
    if len(credit_card) > 0:
        cc_expense = credit_card['amount'].abs().sum()
        if cc_expense > 10000:
            recommendations.append("信用卡支出較大，請確保按時全額繳清以避免利息費用。")
            recommendations.append("Credit card expenses are significant. Ensure you pay the full amount on time to avoid interest charges.")
    
    # Investment recommendations based on SinoPac offerings
    if '股票/Stocks' in df['category'].values:
        recommendations.append("永豐銀行提供外幣定存優惠，考慮利用外幣存款產品來分散您的投資組合。")
        recommendations.append("SinoPac Bank offers preferential rates for foreign currency deposits. Consider using these products to diversify your investment portfolio.")
    
    # Add general recommendations
    recommendations.append("考慮設立自動轉帳以定期將部分收入存入專門的儲蓄或投資帳戶。")
    recommendations.append("Consider setting up automatic transfers to regularly move a portion of your income into dedicated savings or investment accounts.")
    
    recommendations.append("定期檢視對帳單，掌握您的財務狀況，及早發現任何異常交易。")
    recommendations.append("Regularly review your bank statements to stay on top of your financial situation and detect any unusual transactions early.")
    
    return recommendations

def analyze_expenses(file_path):
    """
    Analyzes SinoPac Bank statement and generates charts, categorization, and recommendations.
    
    Args:
        file_path: Path to the SinoPac Bank statement PDF
    """
    # Determine file type and read accordingly
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            df = extract_from_pdf(file_path)
        elif file_ext in ['.csv', '.xlsx', '.xls']:
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        else:
            print(f"Unsupported file format: {file_ext}")
            return None
            
        if df is None or df.empty:
            print("No data could be extracted from the file")
            return None
            
        # Preprocess the dataframe
        df = preprocess_expense_data(df)
        
        # Categorize expenses
        df = categorize_expenses(df)
        
        # Generate visualizations
        create_expense_visualizations(df)
        
        # Generate recommendations
        recommendations = generate_recommendations(df)
        
        return {
            'dataframe': df,
            'recommendations': recommendations
        }
        
    except Exception as e:
        print(f"Error analyzing expenses: {str(e)}")
        traceback.print_exc()
        return None

def main():
    """Main function to run the AI Financial Assistant."""
    print("===== AI 財務助手 / AI Financial Assistant =====")
    print("-----------------------------------------")
    print("Type 'exit' or '退出' to quit (輸入 'exit' 或 '退出' 來結束)")
    
    # Initialize the LLM
    llm = OllamaLLM(model="mistral", temperature=0)  # Lower temperature for more deterministic responses
    
    # Define our tools
    tools = [analyze_bank_statement, open_camera]
    
    # Initialize the agent
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
    )
    
    while True:
        user_input = input("\n輸入您的命令 / Enter your prompt: ")
        
        if user_input.lower() in ["exit", "退出"]:
            print("退出程序。再見！/ Exiting the program. Goodbye!")
            break
        
        try:
            result = agent.run(user_input)
            print("\n結果 / Result:")
            print(result)
        except Exception as e:
            print(f"\n錯誤 / Error: {e}")
            print("\n如果您要分析文件，請確保提供正確的文件路徑。")
            print("If you want to analyze a document, please make sure you provide the correct file path.")

if __name__ == "__main__":
    main()