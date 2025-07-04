log_folder='log/v0-20250507-072010'
# 检查 log_folder 是否存在，如果不存在则创建它
mkdir -p "$log_folder"

python3 src/evaluate.py --test_groupby > "${log_folder}/groupby.log"
# python3 src/evaluate.py --test_where > "${log_folder}/where.log"
python3 src/evaluate.py --test_metric > "${log_folder}/metric.log"
# python3 src/evaluate.py --test_timerange > "${log_folder}/timerange.log"
# python3 src/evaluate.py --test_timeanalyze > "${log_folder}/时间归因.log"
# python3 src/evaluate.py --test_timedims > "${log_folder}/时间维度.log"
# python3 src/evaluate.py --test_orderby > "${log_folder}/orderby.log"
# python3 src/evaluate.py --test_intent 'data/test/pre意图识别数据 - 场景内测试集.csv' > "${log_folder}/intent场景内.log"
# python3 src/evaluate.py --test_intent 'data/test/pre意图识别数据 - 场景外测试集.csv' > "${log_folder}/intent场景外.log"
# python3 src/evaluate.py --test_dims_detail > "${log_folder}/维度详情提参.log"
# python3 src/evaluate.py --test_metric_detail > "${log_folder}/指标详情提参.log"

echo "All tests completed. Logs are saved in ${log_folder}."