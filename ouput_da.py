import os
import pstats
from pstats import SortKey
from collections import defaultdict

def combine_profile_results(directory, lines_to_show=20):
    profile_files = [f for f in os.listdir(directory) if f.endswith('.prof')]
    
    if not profile_files:
        print(f"'{directory}' 디렉토리에 프로파일 파일이 없습니다.")
        return

    combined_stats = defaultdict(lambda: {'ncalls': 0, 'tottime': 0, 'cumtime': 0})

    for profile_file in profile_files:
        full_path = os.path.join(directory, profile_file)
        p = pstats.Stats(full_path)
        p.strip_dirs()
        
        for func, (cc, nc, tt, ct, callers) in p.stats.items():
            combined_stats[func]['ncalls'] += nc
            combined_stats[func]['tottime'] += tt
            combined_stats[func]['cumtime'] += ct

    # 누적 시간으로 정렬
    sorted_stats = sorted(combined_stats.items(), key=lambda x: x[1]['cumtime'], reverse=True)

    print(f"\n통합 프로파일링 결과 상위 {lines_to_show}개:")
    print("ncalls".rjust(10), "tottime".rjust(10), "cumtime".rjust(10), "function")
    print("-" * 60)

    for func, stats in sorted_stats[:lines_to_show]:
        print(f"{stats['ncalls']:10d} {stats['tottime']:10.3f} {stats['cumtime']:10.3f} {func[2]}")

# 사용 예시
profile_directory = "./"  # 프로파일 파일이 저장된 디렉토리 경로
combine_profile_results(profile_directory)
