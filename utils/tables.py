
def convert_report_to_list(report):
    result = [["klasa", "precision", "recall", "f1-score"]]
    report = report.replace("avg / total", "avg")
    report_lines = report.split(sep='\n')
    for line in report_lines:
        if len(line) < 11 or line[10] == ' ':
            continue
        result.append(line.split()[:-1])

    return result


def get_eval_tabel(rows, algorithm_names):
    result = [["", "loss", "categorical_accuracy"]]
    for row, name in zip(rows, algorithm_names):
        row.insert(0, name)
        result.append(row)
    return result
