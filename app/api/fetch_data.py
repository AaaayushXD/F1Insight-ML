import httpx
import os

API_BASE_URL = os.getenv("API_BASE_URL")
if not API_BASE_URL:
    raise ValueError("API_BASE_URL is not set")





async def fetch_data(endpoint, limit=100, offset=0):
    rows = []
    total = 0

    async with httpx.AsyncClient() as client:
        while True:
            url = f"{API_BASE_URL}/{endpoint}.json?limit={limit}&offset={offset}"
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            current_total = int(data['MRData']['total'])
            if total == 0:
                total = current_total
            
            mrdata = data['MRData']
            page_items = []
            
            for key in mrdata.keys():
                if key.endswith('Table') and isinstance(mrdata[key], dict):
                    table = mrdata[key]
                    for table_key in table.keys():
                        if isinstance(table[table_key], list):
                            page_items = table[table_key]
                            rows.extend(page_items)
                            break
            
            if not page_items:
                break
            
            offset += limit
            
            if offset >= total or len(page_items) < limit:
                break

    return {
        "total": total,
        "data": rows
    }