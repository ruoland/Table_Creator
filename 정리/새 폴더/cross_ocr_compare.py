import json
from utils import load_json, save_json, boxes_overlap

def compare_and_merge_results(paddle_results, easy_results):
    merged_results = []
    
    for paddle_item in paddle_results:
        matching_easy_item = find_matching_item(paddle_item, easy_results)
        
        if matching_easy_item:
            if paddle_item['confidence'] > matching_easy_item['confidence']:
                merged_results.append(paddle_item)
            else:
                merged_results.append(matching_easy_item)
        else:
            merged_results.append(paddle_item)
    
    # Add remaining EasyOCR results that didn't match any PaddleOCR result
    for easy_item in easy_results:
        if not any(boxes_overlap(easy_item, merged_item) for merged_item in merged_results):
            merged_results.append(easy_item)
    
    return merged_results

def find_matching_item(item, items_list):
    for other_item in items_list:
        if boxes_overlap(item, other_item):
            return other_item
    return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python compare_results.py <paddle_json> <easy_json> <output_json>")
        sys.exit(1)
    
    paddle_json = sys.argv[1]
    easy_json = sys.argv[2]
    output_json = sys.argv[3]
    
    paddle_results = load_json(paddle_json)
    easy_results = load_json(easy_json)
    
    merged_results = compare_and_merge_results(paddle_results, easy_results)
    save_json(merged_results, output_json)
    print(f"Merged results saved to {output_json}")
    
    # Print results to console
    print("\nMerged Results:")
    for item in merged_results:
        print(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}")