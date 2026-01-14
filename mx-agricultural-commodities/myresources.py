import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.graph_objects as go

# Scraping Tools
def extract_hidden_fields(soup):
    return {
        "__VIEWSTATE": soup.find("input", {"id": "__VIEWSTATE"})["value"],
        "__EVENTVALIDATION": soup.find("input", {"id": "__EVENTVALIDATION"})["value"],
    }
def parse_table(soup):
    table = soup.find("table", id="tblResultados")
    rows = []

    for tr in table.find_all("tr"):
        tds = tr.find_all("td")

        # Skip headers and category rows like "Frutas"
        if len(tds) != 8:
            continue
        if "encabACT2" in tds[0].get("class", []):
            continue

        rows.append([td.get_text(strip=True) for td in tds])

    return rows

def scrape_agricultural(BASE_URL, HEADERS, PARAMS):
    session = requests.Session()
    session.headers.update(HEADERS)

    response = session.post(BASE_URL, params=PARAMS)
    soup = BeautifulSoup(response.text, "lxml")
    all_rows = []
    page = 1

    while True:
        print(f'\rScraping page {page} ...', end='', flush=True)
        all_rows.extend(parse_table(soup))

        # Check pagination text
        pag_text = soup.find("span", id="lblPaginacion").get_text(strip=True)
        current, total = [
            int(x) for x in pag_text.replace("Página", "").split("de")
        ]

        if current >= total:
            break

        hidden = extract_hidden_fields(soup)

        payload = {
            **hidden,
            "__EVENTTARGET": "",
            "__EVENTARGUMENT": "",
            "ibtnSiguiente.x": "1",
            "ibtnSiguiente.y": "1",
        }

        response = session.post(BASE_URL, params=PARAMS, data=payload)
        soup = BeautifulSoup(response.text, "lxml")
        page += 1

    print()
    columns = [
        "Fecha",
        "Presentación",
        "Origen",
        "Destino",
        "Precio Min",
        "Precio Max",
        "Precio Frec",
        "Observaciones",
    ]
    df = pd.DataFrame(all_rows, columns=columns)

    df = df.rename(columns={
        "Fecha": "date",
        "Presentación": "presentation",
        "Origen": "origin",
        "Destino": "destination",
        "Precio Min": "min_price",
        "Precio Max": "max_price",
        "Precio Frec": "avg_price",
        "Observaciones": "notes"
    })
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    price_cols = ["min_price", "max_price", "avg_price"]
    df[price_cols] = df[price_cols].astype(float)

    df = df.drop_duplicates(subset=["date", "presentation", "origin", "destination"], keep="first")
    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)

    return df

def plot_prices_single_origin(df, origin, destinations=None, price_type='avg'):
    """
    Plot prices for a single origin and one or multiple destinations.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns ['date', 'origin', 'destination', 'min_price', 'max_price', 'avg_price']
        origin (str): The origin to filter
        destinations (list or str, optional): Destination(s) to filter. If None, plot all.
        price_type (str): 'min', 'avg', or 'max'
    """
    # Filter by origin
    df_filtered = df[df['origin'] == origin].copy()
    
    # Filter by destination if provided
    if destinations is not None:
        if isinstance(destinations, str):
            destinations = [destinations]
        df_filtered = df_filtered[df_filtered['destination'].isin(destinations)]
    
    # Ensure date is datetime
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    
    fig = go.Figure()
    
    # If multiple destinations, plot only avg
    if destinations is not None and len(destinations) > 1:
        df_grouped = df_filtered.groupby('date')[f'{price_type}_price'].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=df_grouped['date'],
            y=df_grouped[f'{price_type}_price'],
            mode='lines+markers',
            name=f'{price_type.capitalize()} Price (Avg of Destinations)',
            line=dict(color='yellow')
        ))
    else:
        # Plot each destination separately
        for dest in df_filtered['destination'].unique():
            df_dest = df_filtered[df_filtered['destination'] == dest]
            fig.add_trace(go.Scatter(
                x=df_dest['date'],
                y=df_dest[f'{price_type}_price'],
                mode='lines+markers',
                name=f'{dest} ({price_type})'
            ))
    
    # Update layout
    fig.update_layout(
        title=f"{price_type.capitalize()} Prices from {origin}",
        xaxis_title="Date",
        yaxis_title="Price",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='gray'),
        yaxis=dict(showgrid=True, gridcolor='gray')
    )
    
    fig.show()
