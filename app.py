import os
import json
import yfinance as yf
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from tavily import TavilyClient
from datetime import datetime
import re
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize the Groq LLM and Tavily client
model = Groq(id="llama3-70b-8192")
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# -------------------------
# ENHANCED TOOLS
# -------------------------

def validate_ticker(ticker: str) -> bool:
    """Validate stock ticker format"""
    return bool(re.match(r'^[A-Z]{1,5}$', ticker))

def extract_ticker(query: str) -> Optional[str]:
    """Extract potential ticker symbol from query"""
    # Look for all-uppercase words 1-5 letters long
    matches = re.findall(r'\b([A-Z]{1,5})\b', query)
    return matches[0] if matches else None

def extract_company_name(query: str) -> Optional[str]:
    """Extract company name from user query"""
    patterns = [
        r'analysis of (.*?) please',
        r'research on (.*?)$',
        r'information about (.*?)$',
        r'evaluate (.*?)$',
        r'analyze (.*?)$'
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def enhanced_tavily_search(query: str, max_results: int = 5) -> str:
    """Enhanced Tavily search with error handling"""
    try:
        if not query or len(query) < 3:
            return "Error: Search query too short"
            
        response = tavily_client.search(
            query=f"Comprehensive financial research about {query}",
            search_depth="advanced",
            include_raw_content=True,
            max_results=max_results
        )
        
        if not response.get('results'):
            return "Error: No results found"
            
        cleaned_results = []
        for result in response['results'][:max_results]:
            if result.get('content') and result.get('url'):
                cleaned_results.append({
                    'title': result.get('title', 'No title'),
                    'url': result['url'],
                    'content': result['content'][:500] + '...',
                    'published_date': result.get('published_date', 'Unknown')
                })
        
        return json.dumps({
            'query': query,
            'date': datetime.now().isoformat(),
            'results': cleaned_results
        }, indent=2)
        
    except Exception as e:
        return f"Error performing search: {str(e)}"

def get_yfinance_data(ticker: str) -> Dict[str, Any]:
    """Get comprehensive financial data from yfinance"""
    try:
        if not validate_ticker(ticker):
            return {"error": "Invalid ticker symbol"}
            
        stock = yf.Ticker(ticker)
        
        # Basic info
        info = stock.info
        
        # Historical data
        hist = stock.history(period="1y")
        
        # Financial statements
        financials = {
            "income_statement": stock.income_stmt.to_dict() if hasattr(stock, 'income_stmt') else {},
            "balance_sheet": stock.balance_sheet.to_dict() if hasattr(stock, 'balance_sheet') else {},
            "cash_flow": stock.cash_flow.to_dict() if hasattr(stock, 'cash_flow') else {}
        }
        
        # Analyst recommendations
        recommendations = stock.recommendations
        if recommendations is not None and not recommendations.empty:
            recommendations = recommendations.tail(5).to_dict()
        
        # Major holders
        major_holders = stock.major_holders
        if major_holders is not None and not major_holders.empty:
            major_holders = major_holders.to_dict()
        
        # Institutional holders
        institutional_holders = stock.institutional_holders
        if institutional_holders is not None and not institutional_holders.empty:
            institutional_holders = institutional_holders.to_dict()
        
        return {
            "ticker": ticker,
            "company_name": info.get('longName', ticker),
            "current_price": info.get('currentPrice', None),
            "currency": info.get('currency', 'USD'),
            "financials": financials,
            "key_metrics": {
                "pe_ratio": info.get('trailingPE', None),
                "forward_pe": info.get('forwardPE', None),
                "peg_ratio": info.get('pegRatio', None),
                "price_to_book": info.get('priceToBook', None),
                "debt_to_equity": info.get('debtToEquity', None),
                "return_on_equity": info.get('returnOnEquity', None),
                "profit_margins": info.get('profitMargins', None),
                "dividend_yield": info.get('dividendYield', None)
            },
            "analyst_data": {
                "recommendations": recommendations,
                "target_price": info.get('targetMeanPrice', None),
                "recommendation_mean": info.get('recommendationMean', None)
            },
            "ownership": {
                "major_holders": major_holders,
                "institutional_holders": institutional_holders
            },
            "historical_data": hist.to_dict() if hist is not None and not hist.empty else {}
        }
        
    except Exception as e:
        return {"error": f"Failed to fetch yfinance data: {str(e)}"}

# -------------------------
# ENHANCED AGENTS
# -------------------------

research_agent = Agent(
    name="ResearchAgent",
    model=model,
    tools=[enhanced_tavily_search],
    instructions=[
        "You are an investment research specialist. Collect and summarize key qualitative data.",
        "Rules:",
        "1. Use tavily_search tool for research",
        "2. Keep summary under 800 tokens",
        "3. Focus on: market trends, news, analyst opinions, industry outlook",
        "4. Include only top 3 most relevant sources with dates",
        "5. Format with markdown headings",
        "6. Highlight red flags or exceptional positives",
        "7. Never provide investment advice"
    ],
    show_tool_calls=True,
    markdown=True
)

finance_agent = Agent(
    name="FinanceAgent",
    model=model,
    instructions=[
        "You are a certified financial analyst. Analyze quantitative data from yfinance.",
        "Rules:",
        "1. Focus on: valuation ratios, profitability, growth, financial health",
        "2. Compare to industry averages when possible",
        "3. Keep under 800 tokens",
        "4. Use simple language with markdown formatting",
        "5. Highlight concerning metrics in **bold**",
        "6. Include key financial trends from historical data",
        "7. Never provide investment advice"
    ],
    markdown=True
)

analysis_agent = Agent(
    name="AnalysisAgent",
    model=model,
    instructions=[
        "You are a senior investment strategist. Provide objective evaluation.",
        "Rules:",
        "1. Consider both qualitative and quantitative factors",
        "2. Structure with: Strengths, Weaknesses, Risks, Opportunities",
        "3. Keep under 800 tokens",
        "4. Use probabilities not absolutes",
        "5. Disclose data limitations",
        "6. Don't repeat information from other agents",
        "7. No investment recommendations - just analysis"
    ],
    markdown=True
)

editor_agent = Agent(
    name="EditorAgent",
    model=model,
    instructions=[
        "You are a financial editor creating client-ready reports.",
        "Rules:",
        "1. Combine all inputs into one coherent report",
        "2. Maintain original meaning but improve clarity",
        "3. Keep under 1500 tokens",
        "4. Structure:",
        "   - Executive Summary (3 sentences)",
        "   - Key Findings (bulleted)",
        "   - Qualitative Analysis",
        "   - Quantitative Analysis",
        "   - Risk Assessment",
        "   - Final Thoughts",
        "5. Professional but accessible language",
        "6. Include disclaimer and date"
    ],
    markdown=True
)

# -------------------------
# ENHANCED ANALYSIS PIPELINE
# -------------------------

def investment_analysis_pipeline(query: str) -> Dict[str, Any]:
    """Enhanced pipeline with yfinance integration"""
    
    # Extract company and ticker
    company_name = extract_company_name(query)
    ticker = extract_ticker(query)
    
    if not company_name and not ticker:
        return {"error": "Could not identify company or ticker from query"}
    
    # If we have ticker but no company name, use ticker as company name
    if not company_name and ticker:
        company_name = ticker
    
    print(f"\n🔍 Analyzing: {company_name} ({ticker if ticker else 'no ticker'})")
    
    results = {
        "company": company_name,
        "ticker": ticker,
        "date": datetime.now().isoformat(),
        "stages": {}
    }
    
    try:
        # Stage 1: Qualitative Research
        print("🔍 Step 1: Research Agent Running...")
        research_output = research_agent.run(f"Comprehensive financial research on {company_name}")
        results["stages"]["qualitative_research"] = {
            "output": research_output.content,
            "status": "completed"
        }
        
        # Stage 2: Quantitative Analysis (if we have ticker)
        if ticker:
            print("📊 Step 2: Fetching Financial Data...")
            financial_data = get_yfinance_data(ticker)
            
            if "error" in financial_data:
                results["stages"]["quantitative_analysis"] = {
                    "output": financial_data["error"],
                    "status": "failed"
                }
            else:
                print("📈 Step 2a: Finance Agent Analyzing...")
                finance_output = finance_agent.run(
                    f"Analyze these financial metrics: {json.dumps(financial_data, indent=2)}"
                )
                results["stages"]["quantitative_analysis"] = {
                    "output": finance_output.content,
                    "status": "completed",
                    "raw_data": financial_data
                }
        else:
            results["stages"]["quantitative_analysis"] = {
                "output": "No ticker provided - skipping quantitative analysis",
                "status": "skipped"
            }
        
        # Stage 3: Investment Analysis
        print("📈 Step 3: Analysis Agent Running...")
        analysis_input = f"""
        Qualitative Research:
        {research_output.content}
        
        Quantitative Analysis:
        {results['stages']['quantitative_analysis']['output'] if 'quantitative_analysis' in results['stages'] else 'No quantitative data available'}
        """
        
        analysis_output = analysis_agent.run(analysis_input)
        results["stages"]["investment_analysis"] = {
            "output": analysis_output.content,
            "status": "completed"
        }
        
        # Stage 4: Final Report
        print("📝 Step 4: Editor Agent Compiling Report...")
        final_report = editor_agent.run(f"""
        Qualitative Research Summary:
        {research_output.content}
        
        Financial Analysis:
        {results['stages']['quantitative_analysis']['output'] if 'quantitative_analysis' in results['stages'] else 'No financial data available'}
        
        Investment Analysis:
        {analysis_output.content}
        """)
        
        results["final_report"] = final_report.content
        results["status"] = "success"
        
    except Exception as e:
        results["error"] = str(e)
        results["status"] = "failed"
        print(f"Error in analysis pipeline: {str(e)}")
    
    return results

# -------------------------
# REPORT DISPLAY & SAVING
# -------------------------

def display_report(results: Dict[str, Any]) -> None:
    """Display results in a user-friendly format"""
    if "error" in results:
        print(f"\n❌ Error: {results['error']}")
        return
        
    print("\n" + "="*80)
    print(f"📑 FINAL INVESTMENT REPORT: {results['company']} ({results.get('ticker', '')})")
    print("="*80)
    print(results["final_report"])
    print("\n" + "="*80)
    print(f"📅 Report generated on: {results['date']}")
    print("="*80 + "\n")
    
    # Save to file
    filename = f"investment_report_{results['company'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, 'w') as f:
        f.write(results["final_report"])
    print(f"💾 Report saved to: {filename}")

# -------------------------
# MAIN CLI INTERFACE
# -------------------------

def main():
    print("\n💼 ENHANCED INVESTMENT ANALYST AI WITH YFINANCE")
    print("="*80)
    print("This system provides comprehensive investment analysis using:")
    print("- Groq's Llama3-70B for financial reasoning")
    print("- Tavily AI for qualitative research")
    print("- Yahoo Finance (yfinance) for quantitative data")
    print("="*80)
    print("\nExamples of good queries:")
    print("- Research on NVIDIA Corporation (NVDA)")
    print("- Analyze Tesla's financial position (TSLA)")
    print("- Evaluate Apple stock (AAPL)")
    print("\nType 'exit' to quit at any time.\n")
    
    while True:
        try:
            user_input = input("📈 Your investment query: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 Thank you for using the Investment Analyst AI!")
                break
                
            if len(user_input) < 5:
                print("Please provide a more detailed query (at least 5 characters)")
                continue
                
            print("\n⏳ Processing your request...")
            results = investment_analysis_pipeline(user_input)
            display_report(results)
            
        except KeyboardInterrupt:
            print("\n👋 Operation cancelled by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()