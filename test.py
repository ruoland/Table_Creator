import pstats
import os
from pstats import SortKey

def analyze_process_stats(directory):
    # .stats 파일 찾기
    stats_files = [f for f in os.listdir(directory) if f.startswith('process_') and f.endswith('.stats')]
    
    if not stats_files:
        print("프로파일 결과 파일을 찾을 수 없습니다.")
        return

    print(f"{len(stats_files)}개의 프로파일 결과 파일을 찾았습니다.")

    # 각 파일 분석
    for stats_file in stats_files:
        print(f"\n프로세스 파일 분석: {stats_file}")
        stats = pstats.Stats(os.path.join(directory, stats_file))
        
        # 경로 제거 및 누적 시간으로 정렬
        stats.strip_dirs().sort_stats(SortKey.CUMULATIVE)
        
        # 상위 10개 함수 출력
        print("\n상위 10개 함수 (누적 시간 기준):")
        stats.print_stats(10)
        
        # 가장 많이 호출된 함수 출력
        print("\n가장 많이 호출된 함수 (호출 횟수 기준):")
        stats.sort_stats(SortKey.CALLS)
        stats.print_stats(10)
        
        # 특정 함수의 호출자 정보 출력 (예: 'generate_image_and_labels' 함수)
        print("\n'generate_image_and_labels' 함수의 호출자:")
        stats.print_callers('generate_image_and_labels')

# 사용 예
analyze_process_stats('.')  # 현재 디렉토리에서 .stats 파일 찾기
