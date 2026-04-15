import nbformat as nbf

def py_to_ipynb(py_file, ipynb_file):
    with open(py_file, "r", encoding="utf-8") as f:
        code_lines = f.readlines()
    
    notebook = nbf.v4.new_notebook()
    cells = []
    
    cell_code = []
    
    for line in code_lines:
        if line.startswith("# %%"):  # Recognizing code cell markers if used
            if cell_code:
                cells.append(nbf.v4.new_code_cell("".join(cell_code)))
                cell_code = []
        cell_code.append(line)
    
    if cell_code:
        cells.append(nbf.v4.new_code_cell("".join(cell_code)))
    
    notebook.cells = cells

    with open(ipynb_file, "w", encoding="utf-8") as f:
        nbf.write(notebook, f)

# Example usage:
py_to_ipynb("Full.py", "notebook.ipynb")
