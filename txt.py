def print_banner(title=None, line_l=50, line_indent=6, **kwargs):
    if title:
        print(title)
    print("="*line_l)
    for x in kwargs:
        print("-"*line_indent + " {0:s}: {1:s}".format(x, str(kwargs[x])))
    print("="*line_l)

