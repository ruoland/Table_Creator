import pstats
from pstats import SortKey

p = pstats.Stats('output.prof')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)
