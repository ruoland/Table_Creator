import json
from typing import List, Dict, Any

def parse_json_data(json_data: str) -> List[Dict[str, Any]]:
    data = json.loads(json_data)
    return data[0]

def create_timetable(parsed_data: List[Dict[str, Any]]) -> List[List[str]]:
    rows = 11
    cols = 6
    timetable = [["" for _ in range(cols)] for _ in range(rows)]
    
    for item in parsed_data:
        row = item['row'] - 1
        col = item['column'] - 1
        if 0 <= row < rows and 0 <= col < cols:
            timetable[row][col] = item['text']
    
    return timetable

def print_timetable(timetable: List[List[str]]):
    col_widths = [max(len(timetable[row][col]) for row in range(len(timetable))) for col in range(len(timetable[0]))]
    
    for row in timetable:
        print("|", end="")
        for col, cell in enumerate(row):
            print(f" {cell:<{col_widths[col]}} |", end="")
        print()
        print("+", end="")
        for width in col_widths:
            print("-" * (width + 2) + "+", end="")
        print()

def make_timetable(json_data):
    
    parsed_data = parse_json_data(json_data)
    timetable = create_timetable(parsed_data)
    print_timetable(timetable)