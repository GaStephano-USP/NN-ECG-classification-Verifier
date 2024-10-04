
def save_tex_table(rows_list, filename):
    path = "doc/results/"+filename+".tex"
    f = open(path, 'w')
    put_table_to_file(rows_list, f)
    f.close()


def put_table_to_file(rows_list, file):
    table_header = "\\begin{tabular}{|r|" + "l|"*(len(rows_list[0])-1) + "}\n"
    file.write(table_header)
    file.write("  \\hline\n")

    for row in rows_list:
        save_single_row(row, file)

    file.write("\end{tabular}\n")


def save_single_row(row, file):
    row_string = get_row_string(row)
    string_to_save = row_string[:-2] + "\\\\\n"
    file.write(string_to_save)
    file.write("  \\hline\n")


def get_row_string(row):
    row_string = "  "
    for elem in row:
        row_string = row_string + sanitize(elem) + " & "

    return row_string


def sanitize(elem):
    return str(elem).replace("_", " ")
