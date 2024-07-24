import json
from collections import defaultdict

# OCR 결과 데이터

def group_by_rows(data, threshold=10):
    rows = defaultdict(list)
    for item in data:
        y_center = (item['coordinates']['y_min'] + item['coordinates']['y_max']) / 2
        assigned = False
        for row_key in rows:
            if abs(row_key - y_center) < threshold:
                rows[row_key].append(item)
                assigned = True
                break
        if not assigned:
            rows[y_center].append(item)
    return dict(sorted(rows.items()))

def sort_row(row):
    return sorted(row, key=lambda x: x['coordinates']['x_min'])

def create_table(data):
    rows = group_by_rows(data)
    table = []
    for row_key, row_items in rows.items():
        table.append(sort_row(row_items))
    return table

def print_table(table):
    for row in table:
        print(" | ".join(item['text'] for item in row))
        print("-" * (len(row) * 10))

# 표 생성 및 출력
