from numpy import float16, float32, float64, float_


def save_tex_table(rows_list, filename):
    if filename[-4:] != ".tex":
        filename = filename + ".tex"
    path = "./tables/"+filename
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
    if is_float(type(elem)):
        elem = sanitize_float(elem)
    return str(elem).replace("_", " ")


def is_float(t):
    return t == float16 or t == float32 or t == float64 or t == float_ or t == float


def sanitize_float(number):
    return crop_to_three_decimal_places(str(number))


def crop_to_three_decimal_places(number_str):
    dot_position = number_str.find('.')
    if number_str.count('e-') == 0:  # normal dot notation. not 1.0e-4
        return number_str[:dot_position+4]
    else:
        e_position = number_str.find('e')
        return number_str[:dot_position+4] + number_str[e_position:]
