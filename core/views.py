import yfinance as yf
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.template.defaultfilters import title
from core.models import Stock, StockAnalysis, QualitativeAnalysis
from core.utils import CachedTicker
from users.models import Watchlist
from core.trie_instance import stock_trie
from django.http import HttpResponse
from .decorators import yf_ticker_required
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
import json
from django.utils.html import escape
import math

def home(request):
    return render(request, 'core/home.html')

def search_stocks(request):
    search_query = request.POST.get('search_term', '').strip()

    # Check if this is a focus trigger (empty search but triggered)
    if len(search_query) == 0:
        # Show placeholder message when focused but empty
        return HttpResponse('<div class="search-placeholder">Search for a Company</div>')
    elif len(search_query) < 3:
        return HttpResponse('<div class="search-placeholder">Please enter at least 3 characters</div>')
    else:
        # converting it to set then list again to avoid duplicates (ticker and name both are in the trie)
        stocks = list(set(stock_trie.search_prefix(search_query.lower())))
        return render(request, 'core/partials/results.html', {'stocks': stocks})

def _calculate_price_change(history) -> tuple[str, str]:
    if history.empty:
        return "N/A", "grey"
    last_close = history['Close'].iloc[-1]
    first_close = history['Close'].iloc[0]
    change = last_close - first_close

    if change >= 0:
        color = "green"
    else:
        color = "red"

    change_percent = change / first_close * 100
    change_percent = f"▼{abs(change_percent):.2f}%" if change < 0 else f"▲{change_percent:.2f}%"
    change = f"+{change:.2f}" if change >= 0 else f"{change:.2f}"
    return f'<b style="color: {color}" font-family:"Montserrat, sans-serif">{change} ({change_percent})</b>', color

def stock_details(request, ticker):
    try:
        stock = Stock.objects.get(ticker__iexact=ticker)
    except Stock.DoesNotExist:
        return HttpResponse("Stock not found.")

    # Get current price from yfinance
    try:
        yf_ticker = CachedTicker(stock.ticker + ".NS")
        info = yf_ticker.info
        current_price = info.get('currentPrice')
        prev_close = info.get('previousClose')

        # Calculate price change
        if current_price and prev_close:
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            price_change_color = "green" if change >= 0 else "red"
            price_change_symbol = "▲" if change >= 0 else "▼"
        else:
            change = None
            change_percent = None
            price_change_color = "grey"
            price_change_symbol = ""
    except Exception as e:
        print(f"Error fetching current price: {e}")
        current_price = None
        change = None
        change_percent = None
        price_change_color = "grey"
        price_change_symbol = ""

    try:
        saved_analysis = StockAnalysis.objects.get(stock=stock)
        analysis_data = json.loads(saved_analysis.analysis_text)
    except StockAnalysis.DoesNotExist:
        saved_analysis = None
        analysis_data = None

    if request.user.is_authenticated:
        is_in_watchlist = Watchlist.objects.filter(user=request.user, stock=stock).exists()
    else:
        is_in_watchlist = False

    return render(request, 'core/stock_detail.html', {
        'stock': stock,
        'current_price': current_price,
        'change': change,
        'change_percent': change_percent,
        'price_change_color': price_change_color,
        'price_change_symbol': price_change_symbol,
        'saved_analysis': saved_analysis,
        'analysis_data': analysis_data,
        'is_in_watchlist': is_in_watchlist
    })

def get_qualitative_analysis(stock):
    from google import genai
    from google.genai import types
    try:
        client = genai.Client()

        prompt = f"""You are an expert financial analyst tasked with generating a qualitative overview of a company based on publicly available information. Use your knowledge and ability to search the web if necessary.

                **Company:** {stock.name} ({stock.ticker})

                **Instructions:**
                Provide a concise analysis covering the sections below. Use clear, objective language and avoid investment advice. Structure your response using ONLY the following specific Markdown headings and bullet points starting with '* '.

                ## Business Model
                * **Revenue Streams:** Briefly describe how the company makes money (e.g., product sales, subscriptions, services, ads).
                * **Products/Services:** What are its main offerings and their value proposition?
                * **Geography:** Where does it operate and what are its key markets?

                ## Industry and Economic Analysis
                * **Industry Health:** Is the industry growing, stable, or declining? Key trends?
                * **Macroeconomic Factors:** How do broader economic trends like interest rates, inflation, and GDP growth affect the industry and the company?

                ## Competitive Landscape
                * **Market Position:** Is this company a market leader, a challenger, or a niche player? Key competitors?
                * **Competitive Advantage (Moat):** What protects it from competitors? (brand, network effects, switching costs, cost advantages)? Briefly explain why.

                ## Management and Corporate Governance
                * **Leadership Team:** Brief note on the track record and experience of the key executives? Are their interests aligned with shareholders (e.g., do they own stock)?
                * **Transparency/Shareholder Alignment:** Is the company's communication with investors clear, honest, and timely?

                **Output Format:** Strictly adhere to the specified Markdown headings and bullet points. Do not add introductory or concluding sentences outside of these sections."""

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=-1)  # Set thinking budget as needed
            ),
        )
        return response.text
    except Exception as e:
        return HttpResponse(f"Error generating qualitative analysis: {str(e)}", status=500)

def parse_qualitative_analysis(text):
    """
    Parse the qualitative analysis text into structured sections.
    Expected sections: Business Model, Industry and Economic Analysis,
    Competitive Landscape, Management and Corporate Governance
    """
    sections = {
        'business_model': [],
        'industry_analysis': [],
        'competitive_landscape': [],
        'management': []
    }

    current_section = None
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for section headers
        if '## Business Model' in line:
            current_section = 'business_model'
        elif '## Industry and Economic Analysis' in line:
            current_section = 'industry_analysis'
        elif '## Competitive Landscape' in line:
            current_section = 'competitive_landscape'
        elif '## Management and Corporate Governance' in line:
            current_section = 'management'
        # Parse bullet points
        elif line.startswith('* ') and current_section:
            # Remove leading asterisks and clean up
            cleaned_line = line.lstrip('* ').strip()
            # Remove any extra asterisks that might appear
            cleaned_line = cleaned_line.replace('**', '')
            if cleaned_line:
                # If there's a colon, wrap the text before the first colon in <strong>
                if ':' in cleaned_line:
                    idx = cleaned_line.find(':')
                    label = cleaned_line[:idx].strip()
                    rest = cleaned_line[idx:]
                    # Escape label and rest to prevent XSS, then mark safe when rendering in template
                    formatted = f"<strong>{escape(label)}</strong>{escape(rest)}"
                    sections[current_section].append(formatted)
                else:
                    # No colon — escape and keep as plain text
                    sections[current_section].append(escape(cleaned_line))

    return sections

@login_required
def get_qualitative_partial(request, ticker):
    try:
        stock = Stock.objects.get(ticker__iexact=ticker)

        # Try to get existing analysis from database first
        try:
            analysis_obj = QualitativeAnalysis.objects.get(stock=stock)
            # Load the saved analysis
            parsed_analysis = json.loads(analysis_obj.qualitative_analysis)
            print(f"Loaded existing qualitative analysis for {ticker} from database")
        except QualitativeAnalysis.DoesNotExist:
            # No existing analysis, generate new one
            print(f"Generating new qualitative analysis for {ticker}")
            qualitative_analysis = get_qualitative_analysis(stock)
            parsed_analysis = parse_qualitative_analysis(qualitative_analysis)

            # Save to database
            analysis_obj = QualitativeAnalysis.objects.create(
                stock=stock,
                stock_ticker=stock.ticker,
                qualitative_analysis=json.dumps(parsed_analysis)
            )
            print(f"Saved new qualitative analysis for {ticker} to database")

        return render(request, 'core/partials/qualitative_analysis.html', {'qualitative_analysis': parsed_analysis, 'analysis': analysis_obj})
    except Exception as e:
        print(f"Error in get_qualitative_partial: {str(e)}")
        return HttpResponse(f"Error fetching qualitative analysis: {str(e)}", status=500)

@yf_ticker_required
def get_stats_partial(request, ticker, yf_ticker=None):
    try:
        info = yf_ticker.info
        balance_sheet = yf_ticker.balance_sheet
        financials = yf_ticker.financials
        rs = "\u20B9"  # rupee symbol

        # Safe calculation
        current_price = info.get('currentPrice')
        market_cap = info.get('marketCap')
        pe_ratio = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        book_value = info.get('bookValue')
        dividend_yield = info.get('dividendYield')
        trailing_eps = info.get('trailingEps')
        beta = info.get('beta')
        high = info.get('fiftyTwoWeekHigh')
        low = info.get('fiftyTwoWeekLow')
        roce = None
        prev_close = info.get('previousClose')

        ebit = financials.loc['EBIT',financials.columns[0]]
        total_equity = balance_sheet.loc['Stockholders Equity', balance_sheet.columns[0]] if 'Stockholders Equity' in balance_sheet.index else None
        total_debt = info.get('totalDebt')

        if ebit is not None and total_equity is not None and total_debt is not None:
            capital_employed = total_equity + total_debt
            roce = ((ebit / capital_employed) * 100)

        key_stats = {
            "Market Cap": f"{rs} {market_cap/10000000:.2f} Cr" if market_cap and market_cap > 0 else "N/A",
            "Current Price": f"{rs} {current_price:.2f}" if current_price is not None else "N/A",
            "P/E Ratio": f"{pe_ratio:.2f}" if pe_ratio is not None else "N/A",
            "Sector": title(info.get('sector', 'N/A')),
            "Industry": title(info.get('industry', 'N/A')),
            "Forward P/E": f"{forward_pe:.2f}" if forward_pe is not None else "N/A",
            "Book Value": f"{rs} {book_value:.2f}" if book_value is not None else "N/A",
            "Dividend Yield": f"{dividend_yield:.2f}%" if dividend_yield is not None else "0.00%",
            "EPS": f"{rs} {trailing_eps:.2f}" if trailing_eps is not None else "N/A",
            "Beta": f"{beta:.2f}" if beta is not None else "N/A",
            "High/Low": f"{rs} {high:.2f} / {low:.2f}" if high is not None and low is not None else "N/A",
            "ROCE": f"{roce:.2f}%" if roce is not None else "N/A",
        }

        request.session[f'stats_{ticker}'] = key_stats
        return render(request, 'core/partials/stats.html', {'key_stats': key_stats})
    except Exception as e:
        print(f"Error in get_stats_partial: {str(e)}")
        return render(request, 'core/partials/stats.html', {'key_stats': None})

def price_chart(history):
    last_date = history.index.max()
    history_1m = history[history.index >= (last_date - pd.DateOffset(months=1))]
    history_3m = history[history.index >= (last_date - pd.DateOffset(months=3))]
    history_6m = history[history.index >= (last_date - pd.DateOffset(months=6))]
    history_1y = history[history.index >= (last_date - pd.DateOffset(years=1))]
    history_5y = history[history.index >= (last_date - pd.DateOffset(years=5))]

    price_change_1m, color_1m = _calculate_price_change(history_1m)
    price_change_3m, color_3m = _calculate_price_change(history_3m)
    price_change_6m, color_6m = _calculate_price_change(history_6m)
    price_change_1y, color_1y = _calculate_price_change(history_1y)
    price_change_5y, color_5y = _calculate_price_change(history_5y)
    price_change_max, color_max = _calculate_price_change(history)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_1m.index.strftime('%d %b'), y=history_1m['Close'], line=dict(color=color_1m), visible=False))
    fig.add_trace(go.Scatter(x=history_3m.index.strftime('%d %b'), y=history_3m['Close'], line=dict(color=color_3m), visible=False))
    fig.add_trace(go.Scatter(x=history_6m.index.strftime('%d %b'), y=history_6m['Close'], line=dict(color=color_6m), visible=True))
    fig.add_trace(go.Scatter(x=history_1y.index.strftime('%d %b %Y'), y=history_1y['Close'], line=dict(color=color_1y), visible=False))
    fig.add_trace(go.Scatter(x=history_5y.index.strftime('%d %b %Y'), y=history_5y['Close'], line=dict(color=color_5y), visible=False))
    fig.add_trace(go.Scatter(x=history.index.strftime('%d %b %Y'), y=history['Close'], line=dict(color=color_max), visible=False))

    # Apply only to Scatter traces
    fig.update_traces(
        name='',
        mode='lines',
        line=dict(shape='spline', width=2),
        showlegend=False,
        selector=dict(type='scatter'),
        hovertemplate='<b style="font-size: 1.02rem;">₹ %{y:.2f}</b><extra></extra>',
    )

    base_annotation_style = dict(
        align='left',
        x=0, y=1.1,
        showarrow=False,
        xref='paper',
        yref='paper',
        bgcolor="rgba(229, 236, 246, 0)",
        borderwidth=0,
        borderpad=4,
        font=dict(
            family="Montserrat, sans-serif",
            size=16
        )
    )

    annotations_list = [
        dict(**base_annotation_style, text="1M " + price_change_1m, visible=False),
        dict(**base_annotation_style, text="3M " + price_change_3m, visible=False),
        dict(**base_annotation_style, text="6M " + price_change_6m, visible=True),
        dict(**base_annotation_style, text="1Yr " + price_change_1y, visible=False),
        dict(**base_annotation_style, text="5Yr " + price_change_5y, visible=False),
        dict(**base_annotation_style, text="Max " + price_change_max, visible=False)
    ]

    def get_button_args(active_position, size=6):
        trace_visibility = [i == active_position - 1 for i in range(size)]
        # Create a deep copy of annotations and update visibility
        updated_annotations = [ann.copy() for ann in annotations_list]
        for i, ann in enumerate(updated_annotations):
            ann['visible'] = (i == active_position - 1)
        return [{"visible": trace_visibility}, {"annotations": updated_annotations}]

    fig.update_layout(
                      margin=dict(
                          t=0,  # Top margin in pixels
                          b=0,  # Bottom margin in pixels
                          l=40,  # Left margin in pixels
                          r=40  # Right margin in pixels
                      ),
                      height=400,
                      yaxis=dict(
                          title="Price (Rs.)",
                          autorange=True,
                          fixedrange=True,
                      ),
                      hovermode="x",
                      xaxis=dict(
                          title="Date",
                          type="category",
                          nticks=10,
                          fixedrange=False,
                      ),
                      showlegend=False,
                      updatemenus=[
                          dict(
                              type="buttons",
                              direction="right",
                              active=2,
                              x=0.5,
                              y=1.1,
                              xanchor="center",
                              yanchor="top",
                              borderwidth=0,
                              font=dict(family="Montserrat, sans-serif", size=16),
                              bgcolor="rgba(229, 236, 246, 0)",
                              showactive=True,
                              buttons=list([
                                  dict(label="1M",
                                       method="update",
                                       args=get_button_args(1)),  # Show 1M trace
                                  dict(label="3M",
                                       method="update",
                                       args=get_button_args(2)),  # Show 3M trace
                                  dict(label="6M",
                                       method="update",
                                       args=get_button_args(3)),  # Show 6M trace
                                  dict(label="1Yr",
                                       method="update",
                                       args=get_button_args(4)),  # Show 1Y trace
                                  dict(label="5Yr",
                                       method="update",
                                       args=get_button_args(5)),  # Show 5Y trace
                                  dict(label="Max",
                                       method="update",
                                       args=get_button_args(6)),  # Show All trace
                              ]),
                          )
                      ],
                      annotations=annotations_list,

    )
    chart_html = fig.to_html(full_html=False, include_plotlyjs=True, config={'displayModeBar': False})
    return chart_html

@yf_ticker_required
def get_chart_partial(request, ticker, yf_ticker=None):
    try:
        history = yf_ticker.history(period="max")

        chart_html = price_chart(history)
        return render(request, 'core/partials/chart.html', {'chart_html': chart_html})
    except Exception as e:
        print(f"Error in get_chart_partial: {str(e)}")
        return render(request, 'core/partials/chart.html', {'chart_html': None})

def parse_aii_analysis(text):
    sections = {'pros': [], 'cons': [], 'summary': ""}
    current_section = None

    for line in text.split('\n'):
        line = line.strip()
        if "Pros:**" in line or "**Pros:**" in line or line.startswith("Pros"):
            current_section = 'pros'
            continue
        elif "Cons:**" in line or "**Cons:**" in line or line.startswith("Cons"):
            current_section = 'cons'
            continue
        elif "Summary:**" in line or "**Summary:**" in line or line.startswith("Summary"):
            current_section = 'summary'
            continue

        if current_section in ['pros', 'cons'] and line.startswith('* '):
            # Remove the asterisk and process the text
            text_content = line.lstrip('* ')

            # Clean up markdown formatting - remove ** symbols
            text_content = text_content.replace('**', '')

            # Look for the first colon to separate heading from description
            if ':' in text_content:
                colon_index = text_content.index(':')
                bold_part = text_content[:colon_index].strip()
                rest_part = text_content[colon_index + 1:].strip()

                # Format with HTML strong tags
                formatted_text = f"<strong>{bold_part}:</strong> {rest_part}"
            else:
                # If no colon found, make the first few words bold (fallback)
                words = text_content.split()
                if len(words) > 2:
                    bold_part = ' '.join(words[:2])
                    rest_part = ' '.join(words[2:])
                    formatted_text = f"<strong>{bold_part}</strong> {rest_part}"
                else:
                    formatted_text = text_content

            sections[current_section].append(formatted_text)
        elif current_section == 'summary' and line:
            # Clean up markdown formatting for summary as well
            clean_line = line.replace('**', '')
            sections['summary'] = clean_line

    return sections

load_dotenv()

def stock_ai_analysis(stock, key_stats, news):
    from google import genai
    from google.genai import types
    try:
        print(f"Getting Aii Analysis for: {stock.ticker}")
        client = genai.Client()

        stats_str = "\n".join([f"- {key}: {value}" for key, value in key_stats.items() if value != "N/A"])
        news_str = "\n".join([f"- {item['title']} {item['summary']}" for item in news])
        prompt = f"""You are an expert financial analyst providing a summary for an investor.
        Your analysis should be balanced, objective, and based ONLY on the data provided.
        Do not give direct investment advice.

        Company: {stock.name} ({stock.ticker})

        Key Financial Statistics:
        {stats_str}

        Recent News Headlines:
        {news_str}

        Your Task:
        Get additional necessary data from web and Based on the data above and web data, provide a brief analysis covering the following:
        1.  Pros: A few bullish points based on the data you gathered and I provided.
        2.  Cons: A few bearish points or potential risks to consider.
        3.  Summary: A neutral, concluding summary of the company's current position.

        Keep the language clear and concise.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        print(f"Generated Aii analysis for {stock.ticker}:")
        return response.text
    except Exception as e:
        print(f"Error generating Aii analysis: {e}")
        return "Aii analysis is currently unavailable."

def get_analysis_partial(request, ticker):
    # Try to get from POST data first (for backward compatibility)
    stats_json = request.POST.get('stats_json', '{}')
    news_json = request.POST.get('news_json', '[]')
    
    print("Received stats_json:", stats_json)
    print("Received news_json:", news_json)
    
    # If POST data is empty or invalid, try to get from session
    if stats_json == '{}' or news_json == '[]':
        key_stats = request.session.get(f'stats_{ticker}', {})
        news = request.session.get(f'news_{ticker}', [])
        print("Using session data - stats:", key_stats)
        print("Using session data - news:", news)
    else:
        try:
            key_stats = json.loads(stats_json)
            news = json.loads(news_json)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return HttpResponse("Invalid data provided.", status=400)

    if not key_stats and not news:
        return HttpResponse("No data available for analysis.", status=400)

    stock = Stock.objects.get(ticker__iexact=ticker)
    aii_analysis = stock_ai_analysis(stock, key_stats, news)
    parsed_analysis = parse_aii_analysis(aii_analysis)

    analysis_obj, created = StockAnalysis.objects.update_or_create(
        stock=stock,
        defaults={'stock_ticker': stock.ticker, 'analysis_text': json.dumps(parsed_analysis)}
    )
    
    return render(request, 'core/partials/aii_analysis.html',
                  {'analysis_data': parsed_analysis, 'stock': stock})

def analyze_sentiment(text):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    if not text:
        return "No text provided for sentiment analysis."

    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)

    if sentiment_scores['compound'] >= 0.05:
        return "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

@yf_ticker_required
def get_news_partial(request, ticker, yf_ticker=None):
    try:
        news = yf_ticker.news

        if not news:
            print("No news found for ticker:", ticker)
            return render(request, 'core/partials/news.html', {'news': 'no_news'})

        news = sorted(news, key=lambda x: x.get('time', 0), reverse=True)[:6]

        news_with_sentiment = []
        for article in news:
            summary = article['content'].get('summary', '')
            article['sentiment'] = analyze_sentiment(summary)

            # Format date from ISO format to DD/MM/YY
            raw_date = article['content'].get('pubDate', 'No Date found!!')
            formatted_date = raw_date

            if raw_date != 'No Date found!!':
                try:
                    from datetime import datetime
                    # Parse ISO format like "2025-08-24T09:36:04Z"
                    if 'T' in raw_date and 'Z' in raw_date:
                        parsed_date = datetime.fromisoformat(raw_date.replace('Z', '+00:00'))
                        formatted_date = parsed_date.strftime('%d/%m/%y')
                except Exception as e:
                    # If parsing fails, use the raw date
                    formatted_date = raw_date

            news_with_sentiment.append({
                'title': article['content'].get('title', 'No Title'),
                'summary': summary,
                'sentiment': article['sentiment'],
                'link': article['content']['canonicalUrl'].get('url', '#'),
                'date': formatted_date
            })

        request.session[f'news_{ticker}'] = news_with_sentiment

        return render(request, 'core/partials/news.html', {'news': news_with_sentiment})
    except Exception as e:
        print(f"Error in get_news_partial: {e}")
        return render(request, 'core/partials/news.html', {'news': None})

def get_peer_comparison_partial(request, ticker):
    try:
        stock = Stock.objects.get(ticker__iexact=ticker)
        industry = stock.industry

        # Get peers + current stock (total 10 stocks)
        peers = Stock.objects.filter(industry=industry).exclude(ticker=ticker).order_by('-market_cap')[:9]

        # Create a list to store all stocks with their data
        all_stocks_data = []

        # Add peer companies
        for peer in peers:
            try:
                peer_yf = CachedTicker(peer.ticker + ".NS")
                peer_info = peer_yf.info

                market_cap = peer_info.get('marketCap', 0)
                market_cap_display = f"{market_cap/10000000:.2f}" if market_cap and market_cap > 0 else "N/A"

                all_stocks_data.append({
                    "name": peer.name,
                    "ticker": peer.ticker,
                    "current_price": f"{peer_info.get('currentPrice'):.2f}",
                    "market_cap": market_cap_display,
                    "market_cap_raw": market_cap if market_cap else 0,  # For sorting
                    "pe_ratio": f"{peer_info.get('trailingPE'):.2f}" if peer_info.get('trailingPE') is not None else "N/A",
                    "eps": f"{peer_info.get('trailingEps'):.2f}" if peer_info.get('trailingEps') is not None else "N/A",
                    "dividend_yield": f"{peer_info.get('dividendYield'):.2f}%" if peer_info.get('dividendYield') is not None else "0.00%",
                    "beta": f"{peer_info.get('beta'):.2f}" if peer_info.get('beta') is not None else "N/A",
                    "52_week_high": f"{peer_info.get('fiftyTwoWeekHigh'):.2f}" if peer_info.get('fiftyTwoWeekHigh') is not None else "N/A",
                    "52_week_low": f"{peer_info.get('fiftyTwoWeekLow'):.2f}" if peer_info.get('fiftyTwoWeekLow') is not None else "N/A",
                    "is_base": False
                })
            except Exception as e:
                print(f"Error fetching data for peer {peer.ticker}: {e}")
                continue

        # Add the current stock (base stock)
        try:
            base_yf = CachedTicker(stock.ticker + ".NS")
            base_info = base_yf.info

            market_cap = base_info.get('marketCap', 0)
            market_cap_display = f"{market_cap/10000000:.2f}" if market_cap and market_cap > 0 else "N/A"

            all_stocks_data.append({
                "name": stock.name,
                "ticker": stock.ticker,
                "current_price": f"{base_info.get('currentPrice'):.2f}",
                "market_cap": market_cap_display,
                "market_cap_raw": market_cap if market_cap else 0,  # For sorting
                "pe_ratio": f"{base_info.get('trailingPE'):.2f}" if base_info.get('trailingPE') is not None else "N/A",
                "eps": f"{base_info.get('trailingEps'):.2f}" if base_info.get('trailingEps') is not None else "N/A",
                "dividend_yield": f"{base_info.get('dividendYield'):.2f}%" if base_info.get('dividendYield') is not None else "0.00%",
                "beta": f"{base_info.get('beta'):.2f}" if base_info.get('beta') is not None else "N/A",
                "52_week_high": f"{base_info.get('fiftyTwoWeekHigh'):.2f}" if base_info.get('fiftyTwoWeekHigh') is not None else "N/A",
                "52_week_low": f"{base_info.get('fiftyTwoWeekLow'):.2f}" if base_info.get('fiftyTwoWeekLow') is not None else "N/A",
                "is_base": True
            })
        except Exception as e:
            print(f"Error fetching data for base stock {stock.ticker}: {e}")

        # Sort all stocks by market cap (highest first)
        all_stocks_data.sort(key=lambda x: x['market_cap_raw'], reverse=True)

        # Convert to OrderedDict to maintain order in template
        from collections import OrderedDict
        peer_details = OrderedDict()
        for stock_data in all_stocks_data:
            # Remove the raw market cap as it's only needed for sorting
            ticker_key = stock_data['ticker']
            stock_data_copy = stock_data.copy()
            del stock_data_copy['market_cap_raw']
            peer_details[ticker_key] = stock_data_copy

        return render(request, 'core/partials/peer_comparison.html', {'peer_details': peer_details})
    except Exception as e:
        return HttpResponse(f"Error fetching peer comparison data: {str(e)}", status=500)

def format_helper(data, items):
    # Get available years (columns) with month names
    years = [col.strftime('%b %Y') for col in data.columns]
    rs = "₹"

    # Format the data
    formatted_data = []
    for item in items:
        if item['key'] in data.index:
            row_data = {
                'name': item['name'],
                'is_total': item.get('is_total', False),
                'values': []
            }

            for year_col in data.columns:
                value = data.loc[item['key'], year_col]

                # Format the value
                if pd.isna(value) or value == 0:
                    formatted_value = '-'
                elif item["key"] == "Basic EPS":
                    formatted_value = f"{value:.2f}"
                else:
                    # Convert to crores for better readability
                    value_cr = value / 10000000  # Convert to crores
                    formatted_value = f"{abs(value_cr):.2f}"


                row_data['values'].append({
                    'formatted': formatted_value,
                })

            formatted_data.append(row_data)
    return formatted_data, years

def financials_chart(formatted_data, years, show_revenue=False, show_profit=False):
    fig = go.Figure()

    revenue_data = []
    profit_data = []

    # Helper to clean string values like "1,234.56" or "-" into floats
    def parse_value(val_dict):
        v = val_dict.get('formatted', 0)
        if isinstance(v, str):
            v = v.replace(',', '')  # Remove commas if present
            if v == '-' or v == 'N/A':
                return 0.0
            try:
                return float(v)
            except ValueError:
                return 0.0
        return float(v)

    for item in formatted_data:
        if item['name'] == 'Total Revenue':
            revenue_data = [parse_value(v) for v in item['values']]
        elif item['name'] == 'Net Income':
            profit_data = [parse_value(v) for v in item['values']]

    # Add traces
    if show_revenue and revenue_data:
        fig.add_trace(go.Bar(
            x=years,
            y=revenue_data,
            name='Total Revenue',
            marker_color='#0095ff'  # Use your theme color
        ))

    if show_profit and profit_data:
        fig.add_trace(go.Bar(
            x=years,
            y=profit_data,
            name='Net Income',
            marker_color='#2ECC40'
        ))

    # Update layout for a cleaner look
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10, l=10, r=10),
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    chart_html = fig.to_html(full_html=False, include_plotlyjs=True,
                             config={
                                 'displayModeBar': False,
                                 'responsive': True
                             }
    )
    return chart_html

@yf_ticker_required
def get_profit_loss_quarterly_partial(request, ticker, yf_ticker=None):
    try:
        stock = Stock.objects.get(ticker__iexact=ticker)
    except Stock.DoesNotExist:
        return HttpResponse(f'<div class="alert alert-danger">Stock not found.</div>')

    try:
        financials = yf_ticker.quarterly_financials

        if financials.empty:
            return render(request, 'core/partials/profit_loss.html', {'profit_loss_data': None, 'stock': stock})

        # Define the structure for income statement items
        income_statement_items = [
            {'key': 'Total Revenue', 'name': 'Total Revenue'},
            {'key': 'Cost Of Revenue', 'name': 'Cost of Revenue'},
            {'key': 'Gross Profit', 'name': 'Gross Profit'},
            {'key': 'Operating Expense', 'name': 'Operating Expenses'},
            {'key': 'Operating Income', 'name': 'Operating Income'},
            {'key': 'Interest Expense', 'name': 'Interest Expense'},
            {'key': 'Pretax Income', 'name': 'Pretax Income'},
            {'key': 'Tax Provision', 'name': 'Tax Provision'},
            {'key': 'Net Income', 'name': 'Net Income', 'is_total': True},
            {'key': 'Basic EPS', 'name': 'EPS (Rs.)'},
        ]

        formatted_data, years = format_helper(financials, income_statement_items)

        is_chart_update = 'from_chart' in request.GET

        if is_chart_update:
            show_revenue = 'revenue' in request.GET
            show_profit = 'profit' in request.GET
            active_tab = 'chart'
        else:
            show_revenue = True
            show_profit = False
            active_tab = 'financials'

        chart_html = financials_chart(formatted_data, years, show_revenue, show_profit)

        return render(request, 'core/partials/profit_loss.html', {
            'profit_loss_data': formatted_data,
            'years': years,
            'chart_html': chart_html,
            'show_revenue': show_revenue,
            'show_profit': show_profit,
            'active_tab': active_tab,
            'stock': stock,
            'is_annual': False
        })

    except Exception as e:
        print(f"Error fetching quarterly profit & loss data: {e}")
        return HttpResponse(f'<div class="alert alert-danger">Could not load Quarterly Profit & Loss data. Please try refreshing the page.</div>')

@yf_ticker_required
def get_profit_loss_partial(request, ticker, yf_ticker=None):
    try:
        stock = Stock.objects.get(ticker__iexact=ticker)
    except Stock.DoesNotExist:
        return HttpResponse(f'<div class="alert alert-danger">Stock not found.</div>')

    try:
        financials = yf_ticker.financials

        if financials.empty:
            return render(request, 'core/partials/profit_loss.html', {'profit_loss_data': None, 'stock': stock})

        # Define the structure for income statement items
        income_statement_items = [
            {'key': 'Total Revenue', 'name': 'Total Revenue'},
            {'key': 'Cost Of Revenue', 'name': 'Cost of Revenue'},
            {'key': 'Gross Profit', 'name': 'Gross Profit'},
            {'key': 'Operating Expense', 'name': 'Operating Expenses'},
            {'key': 'Operating Income', 'name': 'Operating Income'},
            {'key': 'Interest Expense', 'name': 'Interest Expense'},
            {'key': 'Pretax Income', 'name': 'Pretax Income'},
            {'key': 'Tax Provision', 'name': 'Tax Provision'},
            {'key': 'Net Income', 'name': 'Net Income', 'is_total': True},
            {'key': 'Basic EPS', 'name': 'EPS (Rs.)'},
        ]

        formatted_data, years = format_helper(financials, income_statement_items)

        is_chart_update = 'from_chart' in request.GET

        if is_chart_update:
            show_revenue = 'revenue' in request.GET
            show_profit = 'profit' in request.GET
            active_tab = 'chart'
        else:
            show_revenue = True
            show_profit = False
            active_tab = 'financials'

        chart_html = financials_chart(formatted_data, years, show_revenue, show_profit)

        return render(request, 'core/partials/profit_loss.html', {
            'profit_loss_data': formatted_data,
            'years': years,
            'chart_html': chart_html,
            'show_revenue': show_revenue,
            'show_profit': show_profit,
            'active_tab': active_tab,
            'stock': stock,
            'is_annual': True
        })

    except Exception as e:
        print(f"Error fetching profit & loss data: {e}")
        return HttpResponse(f'<div class="alert alert-danger">Could not load Profit & Loss data. Please try refreshing the page.</div>')

def balance_sheet_chart(formatted_data, years, show_assets=False, show_liabilities=False, show_equity=False):
    fig = go.Figure()

    assets_data = []
    liabilities_data = []
    equity_data = []

    def parse_value(val_dict):
        v = val_dict.get('formatted', 0)
        if isinstance(v, str):
            v = v.replace(',', '')  # Remove commas if present
            if v == '-' or v == 'N/A':
                return 0.0
            try:
                return float(v)
            except ValueError:
                return 0.0
        return float(v)

    for item in formatted_data:
        if item['name'] == 'Total Assets':
            assets_data = [parse_value(v) for v in item['values']]
        elif item['name'] == 'Total Liabilities':
            liabilities_data = [parse_value(v) for v in item['values']]
        elif item['name'] == 'Total Equity':
            equity_data = [parse_value(v) for v in item['values']]

    if show_assets and assets_data:
        fig.add_trace(go.Bar(
            x=years,
            y=assets_data,
            name='Total Assets',
            marker_color='#0095ff'  # Use your theme color
        ))
    if show_liabilities and liabilities_data:
        fig.add_trace(go.Bar(
            x=years,
            y=liabilities_data,
            name='Total Liabilities',
            marker_color='#2ECC40'
        ))
    if show_equity and equity_data:
        fig.add_trace(go.Bar(
            x=years,
            y=equity_data,
            name='Total Equity',
            marker_color='#F5E727'
        ))

    # Update layout for a cleaner look
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10, l=10, r=10),
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    chart_html = fig.to_html(full_html=False, include_plotlyjs=True,
                             config={
                                 'displayModeBar': False,
                                 'responsive': True
                             }
    )
    return chart_html

@yf_ticker_required
def get_balance_sheet_partial(request, ticker, yf_ticker=None):
    try:
        stock = Stock.objects.get(ticker__iexact=ticker)
    except Stock.DoesNotExist:
        return HttpResponse(f'<div class="alert alert-danger">Stock not found.</div>')

    try:
        balance_sheet = yf_ticker.balance_sheet

        if balance_sheet.empty:
            return render(request, 'core/partials/balance_sheet.html', {'balance_sheet_data': None, 'stock': stock})

        # Define the structure for balance sheet items
        balance_sheet_items = [
            # Assets Section
            {'key': 'Current Assets', 'name': 'Current Assets'},
            {'key': 'Cash And Cash Equivalents', 'name': 'Cash & Cash Equivalents'},
            {'key': 'Accounts Receivable', 'name': 'Accounts Receivable'},
            {'key': 'Inventory', 'name': 'Inventory'},
            {'key': 'Other Current Assets', 'name': 'Other Current Assets'},
            {'key': 'Total Current Assets', 'name': 'Total Current Assets'},

            {'key': 'Non Current Assets', 'name': 'Non-Current Assets'},
            {'key': 'Properties', 'name': 'Property, Plant & Equipment'},
            {'key': 'Goodwill', 'name': 'Goodwill'},
            {'key': 'Other Non Current Assets', 'name': 'Other Non-Current Assets'},
            {'key': 'Total Non Current Assets', 'name': 'Total Non-Current Assets'},

            {'key': 'Total Assets', 'name': 'Total Assets', 'is_total': True},

            # Liabilities Section
            {'key': 'Current Liabilities', 'name': 'Current Liabilities'},
            {'key': 'Accounts Payable', 'name': 'Accounts Payable'},
            {'key': 'Current Debt', 'name': 'Short-term Debt'},
            {'key': 'Other Current Liabilities', 'name': 'Other Current Liabilities'},
            {'key': 'Total Current Liabilities', 'name': 'Total Current Liabilities', 'is_total': True},

            {'key': 'Non Current Liabilities', 'name': 'Non-Current Liabilities'},
            {'key': 'Long Term Debt', 'name': 'Long-term Debt'},
            {'key': 'Other Non Current Liabilities', 'name': 'Other Non-Current Liabilities'},
            {'key': 'Total Non Current Liabilities Net Minority Interest', 'name': 'Total Non-Current Liabilities',},

            {'key': 'Total Liabilities Net Minority Interest', 'name': 'Total Liabilities', 'is_total': True},

            # Equity Section
            {'key': 'Stockholders Equity', 'name': 'Shareholders\' Equity'},
            {'key': 'Common Stock', 'name': 'Share Capital'},
            {'key': 'Retained Earnings', 'name': 'Retained Earnings'},
            {'key': 'Other Stockholder Equity', 'name': 'Other Equity'},
            {'key': 'Total Stockholders Equity', 'name': 'Total Shareholders\' Equity'},

            {'key': 'Total Equity Gross Minority Interest', 'name': 'Total Equity', 'is_total': True},
        ]

        formatted_data, years = format_helper(balance_sheet, balance_sheet_items)

        is_chart_update = 'from_chart' in request.GET
        if is_chart_update:
            show_assets = 'assets' in request.GET
            show_liabilities = 'liabilities' in request.GET
            show_equity = 'equity' in request.GET
            active_tab = 'chart'
        else:
            show_assets = True
            show_liabilities = True
            show_equity = False
            active_tab = 'financials'

        chart_html = balance_sheet_chart(formatted_data, years, show_assets, show_liabilities, show_equity)

        return render(request, 'core/partials/balance_sheet.html', {
            'balance_sheet_data': formatted_data,
            'years': years,
            'chart_html': chart_html,
            'show_assets': show_assets,
            'show_liabilities': show_liabilities,
            'show_equity': show_equity,
            'active_tab': active_tab,
            'stock': stock,
        })

    except Exception as e:
        print(f"Error fetching balance sheet data: {e}")
        return HttpResponse(f'<div class="alert alert-danger">Could not load Balance Sheet data. Please try refreshing the page.</div>')

def cash_flow_chart(formatted_data, years, show_operating_cash_flow=False, show_investing_cash_flow=False, show_financing_cash_flow=False, show_net_cash_flow=False):
    fig = go.Figure()

    operating_cash_flow = []
    investing_cash_flow = []
    financing_cash_flow = []
    net_cash_flow = []

    def parse_value(val_dict):
        v = val_dict.get('formatted', 0)
        if isinstance(v, str):
            v = v.replace(',', '')  # Remove commas if present
            if v == '-' or v == 'N/A':
                return 0.0
            try:
                return float(v)
            except ValueError:
                return 0.0
        return float(v)

    for item in formatted_data:
        if item['name'] == 'Operating Cash Flow':
            operating_cash_flow = [parse_value(v) for v in item['values']]
        elif item['name'] == 'Investing Cash Flow':
            investing_cash_flow = [parse_value(v) for v in item['values']]
        elif item['name'] == 'Financing Cash Flow':
            financing_cash_flow = [parse_value(v) for v in item['values']]
        elif item['name'] == 'Net Cash Flow':
            net_cash_flow = [parse_value(v) for v in item['values']]

    if show_operating_cash_flow and operating_cash_flow:
        fig.add_trace(go.Bar(
            x=years,
            y=operating_cash_flow,
            name='Operating Cash Flow',
            marker_color='#0095ff'
        ))
    if show_investing_cash_flow and investing_cash_flow:
        fig.add_trace(go.Bar(
            x=years,
            y=investing_cash_flow,
            name='Investing Cash Flow',
            marker_color='#2ECC40'
        ))
    if show_financing_cash_flow and financing_cash_flow:
        fig.add_trace(go.Bar(
            x=years,
            y=financing_cash_flow,
            name='Financing Cash Flow',
            marker_color='#F5E727'
        ))
    if show_net_cash_flow and net_cash_flow:
        fig.add_trace(go.Bar(
            x=years,
            y=net_cash_flow,
            name='Net Cash Flow',
            marker_color='#FF6347'
        ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10, l=10, r=10),
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    chart_html = fig.to_html(full_html=False, include_plotlyjs=True,
                             config={
                                 'displayModeBar': False,
                                 'responsive': True
                             }
    )
    return chart_html

@yf_ticker_required
def get_cash_flow_partial(request, ticker, yf_ticker=None):
    try:
        stock = Stock.objects.get(ticker__iexact=ticker)
    except Stock.DoesNotExist:
        return HttpResponse(f'<div class="alert alert-danger">Stock not found.</div>')

    try:
        cash_flow = yf_ticker.cashflow

        if cash_flow.empty:
            return render(request, 'core/partials/cash_flow.html', {'cash_flow_data': None, 'stock': stock})

        # Define the structure for cash flow items
        cash_flow_items = [
            # Operating Activities
            {'key': 'Operating Cash Flow', 'name': 'Operating Cash Flow', 'is_total': True},
            {'key': 'Net Income', 'name': 'Net Income'},
            {'key': 'Depreciation And Amortization', 'name': 'Depreciation & Amortization'},
            {'key': 'Deferred Tax', 'name': 'Deferred Tax'},
            {'key': 'Stock Based Compensation', 'name': 'Stock-Based Compensation'},
            {'key': 'Change In Working Capital', 'name': 'Change in Working Capital'},
            {'key': 'Change In Accounts Receivable', 'name': 'Change in Accounts Receivable'},
            {'key': 'Change In Inventory', 'name': 'Change in Inventory'},
            {'key': 'Change In Accounts Payable', 'name': 'Change in Accounts Payable'},
            {'key': 'Change In Other Working Capital', 'name': 'Change in Other Working Capital'},
            {'key': 'Other Operating Cash Flow Activities', 'name': 'Other Operating Activities'},

            # Investing Activities
            {'key': 'Investing Cash Flow', 'name': 'Investing Cash Flow', 'is_total': True},
            {'key': 'Capital Expenditures', 'name': 'Capital Expenditures'},
            {'key': 'Net Business Purchase And Sale', 'name': 'Business Acquisitions/Disposals'},
            {'key': 'Net Investment Purchase And Sale', 'name': 'Investment Purchase/Sale'},
            {'key': 'Net PPE Purchase And Sale', 'name': 'Property, Plant & Equipment'},
            {'key': 'Other Investing Cash Flow Activities', 'name': 'Other Investing Activities'},

            # Financing Activities
            {'key': 'Financing Cash Flow', 'name': 'Financing Cash Flow', 'is_total': True},
            {'key': 'Net Long Term Debt Issuance', 'name': 'Net Long-term Debt Issuance'},
            {'key': 'Net Short Term Debt Issuance', 'name': 'Net Short-term Debt Issuance'},
            {'key': 'Net Common Stock Issuance', 'name': 'Net Common Stock Issuance'},
            {'key': 'Repurchase Of Capital Stock', 'name': 'Share Repurchases'},
            {'key': 'Cash Dividends Paid', 'name': 'Dividends Paid'},
            {'key': 'Other Financing Cash Flow Activities', 'name': 'Other Financing Activities'},

            # Net Cash Flow
            {'key': 'End Cash Position', 'name': 'End Cash Position', 'is_total': True},
            {'key': 'Beginning Cash Position', 'name': 'Beginning Cash Position'},
            {'key': 'Changes In Cash', 'name': 'Net Cash Flow', 'is_total': True},
            {'key': 'Free Cash Flow', 'name': 'Free Cash Flow', 'is_total': True},
        ]

        formatted_data, years = format_helper(cash_flow, cash_flow_items)

        is_chart_update = 'from_chart' in request.GET

        if is_chart_update:
            show_operating_cash_flow = 'operating' in request.GET
            show_investing_cash_flow = 'investing' in request.GET
            show_financing_cash_flow = 'financing' in request.GET
            show_net_cash_flow = 'net' in request.GET
            active_tab = 'chart'
        else:
            show_operating_cash_flow = False
            show_investing_cash_flow = False
            show_financing_cash_flow = False
            show_net_cash_flow = True
            active_tab = 'financials'

        chart_html = cash_flow_chart(formatted_data, years, show_operating_cash_flow, show_investing_cash_flow, show_financing_cash_flow, show_net_cash_flow)

        return render(request, 'core/partials/cash_flow.html', {
            'cash_flow_data': formatted_data,
            'years': years,
            'chart_html': chart_html,
            'show_operating_cash_flow': show_operating_cash_flow,
            'show_investing_cash_flow': show_investing_cash_flow,
            'show_financing_cash_flow': show_financing_cash_flow,
            'show_net_cash_flow': show_net_cash_flow,
            'active_tab': active_tab,
            'stock': stock,
        })

    except Exception as e:
        print(f"Error fetching cash flow data: {e}")
        return HttpResponse(f'<div class="alert alert-danger">Could not load Cash Flow data. Please try refreshing the page.</div>')

@yf_ticker_required
def get_ratios_partial(request, ticker, yf_ticker=None):
    try:
        info = yf_ticker.info
        balance_sheet = yf_ticker.balance_sheet
        financials = yf_ticker.financials
        quarters = yf_ticker.quarterly_financials
        rs = "\u20B9"  # rupee symbol

        # Get valuation ratios from info
        trailing_pe_ratio = info.get('trailingPE')
        forward_pe_ratio = info.get('forwardPE')
        price_to_book = info.get('priceToBook')
        price_to_sales = None
        dividend_yield = info.get('dividendYield')
        ev_ebitda = info.get('enterpriseToEbitda')

        # Get profitability ratios from info
        return_on_equity = None
        return_on_assets = None
        gross_profit_margin = None
        operating_margin = info.get('operatingMargins')
        net_profit_margin = info.get('profitMargins')
        earnings_per_share = info.get('trailingEps')

        # Get leverage ratios from info
        debt_to_equity = info.get('debtToEquity')
        interest_coverage_ratio = None

        # Calculate liquidity ratios from balance sheet
        current_ratio = None
        quick_ratio = None
        debt_to_assets = None

        if not balance_sheet.empty:
            try:
                market_cap = info.get('marketCap')
                total_revenue = financials.loc['Total Revenue', financials.columns[0]] if 'Total Revenue' in financials.index else None
                current_assets = balance_sheet.loc['Current Assets', balance_sheet.columns[0]] if 'Current Assets' in balance_sheet.index else None
                current_liabilities = balance_sheet.loc['Current Liabilities', balance_sheet.columns[0]] if 'Current Liabilities' in balance_sheet.index else None
                inventory = balance_sheet.loc['Inventory', balance_sheet.columns[0]] if 'Inventory' in balance_sheet.index else None
                total_assets = balance_sheet.loc['Total Assets', balance_sheet.columns[0]] if 'Total Assets' in balance_sheet.index else None
                total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest', balance_sheet.columns[0]] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None
                net_income = financials.loc['Net Income', financials.columns[0]] if 'Net Income' in financials.index else None
                avg_stockholder_equity = balance_sheet.loc['Stockholders Equity', balance_sheet.columns[0]] if 'Stockholders Equity' in balance_sheet.index else None
                ebit = financials.loc['EBIT', financials.columns[0]] if 'EBIT' in financials.index else None
                interest_expense = financials.loc['Interest Expense', financials.columns[0]] if 'Interest Expense' in financials.index else None
                gross_profit_quarter = quarters.loc['Gross Profit', quarters.columns[0]] if 'Gross Profit' in quarters.index else None
                revenue_quarter = quarters.loc['Total Revenue', quarters.columns[0]] if 'Total Revenue' in quarters.index else None

                if market_cap is not None and total_revenue is not None and total_revenue != 0:
                    price_to_sales = market_cap / total_revenue

                if current_assets is not None and current_liabilities is not None and current_liabilities != 0:
                    current_ratio = current_assets / current_liabilities

                if current_assets is not None and inventory is not None and current_liabilities is not None and current_liabilities != 0:
                    quick_ratio = (current_assets - inventory) / current_liabilities

                if total_assets is not None and total_liabilities is not None and total_assets != 0:
                    debt_to_assets = total_liabilities / total_assets

                if net_income is not None and avg_stockholder_equity is not None:
                    return_on_equity = net_income / avg_stockholder_equity

                if net_income is not None and total_assets is not None:
                    return_on_assets = net_income / total_assets

                if gross_profit_quarter is not None and revenue_quarter is not None:
                    gross_profit_margin = gross_profit_quarter / revenue_quarter

                if ebit is not None and interest_expense is not None:
                    interest_coverage_ratio = ebit / interest_expense
            except Exception as e:
                print(f"Error calculating ratios from balance sheet: {e}")

        # Format ratios for display with formulas
        formatted_ratios = {
            # Valuation Ratios
            'Trailing P/E Ratio': {
                'value': f"{trailing_pe_ratio:.2f}" if trailing_pe_ratio is not None and trailing_pe_ratio > 0 else "N/A",
                'formula': "Market Price per Share / Earnings per Share (EPS)"
            },
            'Forward P/E Ratio': {
                'value': f"{forward_pe_ratio:.2f}" if forward_pe_ratio is not None and forward_pe_ratio > 0 else "N/A",
                'formula': "Current Stock Price / Forecasted Earnings per Share"
            },
            'Price to Book': {
                'value': f"{price_to_book:.2f}" if price_to_book is not None and price_to_book > 0 else "N/A",
                'formula': "Market Price per Share / Book Value per Share"
            },
            'Price to Sales': {
                'value': f"{price_to_sales:.2f}" if price_to_sales is not None and price_to_sales > 0 else "N/A",
                'formula': "Market Cap / Total Revenue"
            },
            'EV/EBITDA': {
                'value': f"{ev_ebitda:.2f}" if ev_ebitda is not None and ev_ebitda > 0 else "N/A",
                'formula': "Enterprise Value / Earnings Before Interest, Taxes, Depreciation & Amortization"
            },
            'Dividend Yield': {
                'value': f"{dividend_yield * 100:.2f}%" if dividend_yield is not None else "0.00%",
                'formula': "(Annual Dividends per Share / Price per Share) × 100"
            },

            # Profitability Ratios
            'Return on Equity (ROE)': {
                'value': f"{return_on_equity * 100:.2f}%" if return_on_equity is not None else "N/A",
                'formula': "(Net Income / Shareholders' Equity) × 100"
            },
            'Return on Assets (ROA)': {
                'value': f"{return_on_assets * 100:.2f}%" if return_on_assets is not None else "N/A",
                'formula': "(Net Income / Total Assets) × 100"
            },
            'Earnings Per Share (EPS)': {
                'value': f"{rs} {earnings_per_share:.2f}" if earnings_per_share is not None else "N/A",
                'formula': "Net Income / Total Outstanding Shares"
            },
            'Gross Profit Margin': {
                'value': f"{gross_profit_margin * 100:.2f}%" if gross_profit_margin is not None else "N/A",
                'formula': "((Revenue - Cost of Goods Sold) / Revenue) × 100"
            },
            'Operating Margin': {
                'value': f"{operating_margin * 100:.2f}%" if operating_margin is not None else "N/A",
                'formula': "(Operating Income / Revenue) × 100"
            },
            'Net Profit Margin': {
                'value': f"{net_profit_margin * 100:.2f}%" if net_profit_margin is not None else "N/A",
                'formula': "(Net Income / Revenue) × 100"
            },

            # Liquidity Ratios
            'Current Ratio': {
                'value': f"{current_ratio:.2f}" if current_ratio is not None else "N/A",
                'formula': "Current Assets / Current Liabilities"
            },
            'Quick Ratio': {
                'value': f"{quick_ratio:.2f}" if quick_ratio is not None else "N/A",
                'formula': "(Current Assets - Inventory) / Current Liabilities"
            },

            # Leverage Ratios
            'Debt to Equity': {
                'value': f"{debt_to_equity / 100:.2f}" if debt_to_equity is not None else "N/A",
                'formula': "Total Debt / Shareholders' Equity"
            },
            'Debt to Assets': {
                'value': f"{debt_to_assets:.2f}" if debt_to_assets is not None else "N/A",
                'formula': "Total Liabilities / Total Assets"
            },
            'Interest Coverage': {
                'value': f"{interest_coverage_ratio:.2f}" if interest_coverage_ratio is not None else "N/A",
                'formula': "EBIT / Interest Expense"
            },
        }

        return render(request, 'core/partials/ratios.html', {'ratios': formatted_ratios})
    except Exception as e:
        print(f"Error fetching ratios data: {e}")
        import traceback
        traceback.print_exc()
        return render(request, 'core/partials/ratios.html', {'ratios': None})

@yf_ticker_required
def get_shareholding_partial(request, ticker, yf_ticker=None):
    try:
        # 1. Fetch Data
        major = yf_ticker.major_holders
        inst_holders = yf_ticker.institutional_holders
        mf_holders = yf_ticker.mutualfund_holders

        # Defaults
        insider_pct = 0.0
        institution_pct = 0.0

        # 2. Process Major Holders (Fixed for .NS tickers)
        if major is not None and not major.empty:
            for index, row in major.iterrows():

                k = str(index)  # Key (e.g., 'insidersPercentHeld')

                try:
                    v = row.iloc[0]  # Value (e.g., 0.73)
                except (IndexError, AttributeError):
                    continue

                # Clean the value
                if isinstance(v, str) and '%' in v:
                    v = v.replace('%', '')
                    try:
                        val_float = float(v) / 100.0
                    except ValueError:
                        continue
                else:
                    try:
                        val_float = float(v)
                    except (ValueError, TypeError):
                        continue

                # --- LOGIC FIX HERE ---
                k_lower = k.lower()

                if 'insider' in k_lower or 'promoter' in k_lower:
                    insider_pct = val_float

                # Check for 'institution', ensure NOT 'float' and NOT 'count'
                elif 'institution' in k_lower and 'float' not in k_lower and 'count' not in k_lower:
                    institution_pct = val_float

        # 3. Calculate Public Holding
        # Public = 1.0 - (Insiders + Institutions)
        public_pct = 1.0 - (insider_pct + institution_pct)

        # Safety check
        if public_pct < 0:
            public_pct = 0

        # 4. Create Donut Chart
        labels = ['Promoters/Insiders', 'Institutions', 'Public/Others']
        values = [insider_pct, institution_pct, public_pct]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.5,
            marker=dict(colors=['#0095ff', '#2ECC40', '#FF851B']),
            textinfo='percent',
            hoverinfo='label+percent',
            textposition='inside',
        )])

        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        chart_html = fig.to_html(full_html=False, include_plotlyjs=True,
                                 config={'displayModeBar': False, 'responsive': True})

        # 5. Process Tables (Helper function)
        def process_holders(df):
            if df is None or df.empty:
                return None
            data = []
            try:
                for idx, row in df.head(5).iterrows():
                    data.append({
                        'holder': row['Holder'],
                        'pct': f"{row['% Out'] * 100:.2f}%" if '% Out' in row else 'N/A'
                    })
                return data
            except Exception:
                return None

        top_inst = process_holders(inst_holders)
        top_mf = process_holders(mf_holders)

        return render(request, 'core/partials/shareholding.html', {
            'chart_html': chart_html,
            'top_institutions': top_inst,
            'top_mutual_funds': top_mf,
            'insider_pct': f"{insider_pct * 100:.2f}%",
            'inst_pct': f"{institution_pct * 100:.2f}%",
            'public_pct': f"{public_pct * 100:.2f}%"
        })

    except Exception as e:
        print(f"Error fetching shareholding data: {e}")
        return HttpResponse('<div class="alert alert-warning">Shareholding pattern data not available.</div>')


@yf_ticker_required
def get_valuation_partial(request, ticker, yf_ticker=None):
    try:
        info = yf_ticker.info
        rs = "\u20B9"

        current_price = info.get('currentPrice')
        eps = info.get('trailingEps')
        book_value = info.get('bookValue')
        shares_outstanding = info.get('sharesOutstanding')

        # --- 1. ROBUST FCF & NET INCOME RETRIEVAL ---
        fcf = info.get('freeCashflow')
        net_income = info.get('netIncomeToCommon')

        # Try to fetch from dataframes if info is missing
        try:
            cashflow_df = yf_ticker.cashflow
            financials_df = yf_ticker.financials

            # Get FCF from dataframe if missing
            if fcf is None and not cashflow_df.empty:
                if 'Free Cash Flow' in cashflow_df.index:
                    fcf = cashflow_df.loc['Free Cash Flow'].iloc[0]
                elif 'Operating Cash Flow' in cashflow_df.index and 'Capital Expenditures' in cashflow_df.index:
                    ocf = cashflow_df.loc['Operating Cash Flow'].iloc[0]
                    capex = cashflow_df.loc['Capital Expenditures'].iloc[0]
                    fcf = ocf + capex

            # Get Net Income from dataframe if missing
            if net_income is None and not financials_df.empty:
                if 'Net Income' in financials_df.index:
                    net_income = financials_df.loc['Net Income'].iloc[0]

        except Exception as e:
            print(f"Error extracting financial data: {e}")

        # --- 2. Determine DCF Base Metric (The Fix) ---
        # We prefer FCF, but if it's negative/missing, we fallback to Net Income
        dcf_base_value = 0
        metric_used = "N/A"

        if fcf is not None and fcf > 0:
            dcf_base_value = fcf
            metric_used = "Free Cash Flow"
        elif net_income is not None and net_income > 0:
            dcf_base_value = net_income
            metric_used = "Net Income (Earnings)"

        # --- 3. Graham Number Logic ---
        graham_number = None
        valuation_status = "N/A"
        difference_pct = 0
        graham_color = "grey"

        if eps is not None and book_value is not None and eps > 0 and book_value > 0:
            product = 22.5 * eps * book_value
            graham_number = math.sqrt(product)

            if current_price:
                if current_price < graham_number:
                    valuation_status = "Undervalued"
                    difference_pct = ((graham_number - current_price) / current_price) * 100
                    graham_color = "green"
                else:
                    valuation_status = "Overvalued"
                    difference_pct = ((current_price - graham_number) / graham_number) * 100
                    graham_color = "red"

        # --- 4. DCF Calculation Logic ---
        dcf_price = None

        try:
            growth_rate_input = float(request.GET.get('growth_rate', 10))
        except ValueError:
            growth_rate_input = 10

        # Calculate DCF if we have a valid positive base value
        if dcf_base_value > 0 and shares_outstanding:
            growth_rate = growth_rate_input / 100.0
            discount_rate = 0.10
            terminal_growth_rate = 0.02
            years = 10

            future_vals = []

            for i in range(1, years + 1):
                projected_val = dcf_base_value * ((1 + growth_rate) ** i)
                discounted_val = projected_val / ((1 + discount_rate) ** i)
                future_vals.append(discounted_val)

            val_final_year = dcf_base_value * ((1 + growth_rate) ** years)
            terminal_value = (val_final_year * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
            terminal_value_discounted = terminal_value / ((1 + discount_rate) ** years)

            total_dcf_value = sum(future_vals) + terminal_value_discounted
            dcf_price = total_dcf_value / shares_outstanding

        return render(request, 'core/partials/valuation.html', {
            'stock': {'ticker': ticker},
            'graham_number': graham_number,
            'current_price': current_price,
            'eps': eps,
            'book_value': book_value,
            'valuation_status': valuation_status,
            'difference_pct': f"{abs(difference_pct):.2f}%",
            'color': graham_color,
            'rs': rs,
            'dcf_price': dcf_price,
            'growth_rate_input': growth_rate_input,
            'show_dcf': dcf_price is not None,
            'metric_used': metric_used
        })

    except Exception as e:
        print(f"Error calculating valuation: {e}")
        return HttpResponse('<div class="alert alert-warning">Valuation data not available.</div>')