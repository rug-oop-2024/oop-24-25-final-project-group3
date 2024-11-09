import pydoc
import os

# List of Python scripts/modules to generate documentation for
scripts = [
    'path/to/script1',  # Without .py extension
    'path/to/script2',
    'path/to/script3',
    # Add more scripts as needed
]

# Directory to store the generated documentation files temporarily
temp_doc_dir = './temp_docs'
os.makedirs(temp_doc_dir, exist_ok=True)

# Generate documentation for each script
for script in scripts:
    module_name = script.replace('/', '.').rstrip('.py')
    pydoc.writedoc(module_name)
    # Move generated file to the temporary docs directory
    os.rename(f"{module_name}.html", os.path.join(temp_doc_dir, f"{module_name}.html"))

# Combine all the generated HTML files into one document
combined_doc_path = 'combined_documentation.html'
with open(combined_doc_path, 'w', encoding='utf-8') as combined_doc:
    combined_doc.write('<html><head><title>Combined Documentation</title></head><body>')
    combined_doc.write('<h1>Combined Documentation</h1>')
    combined_doc.write('<h2>Table of Contents</h2>')
    combined_doc.write('<ul>')

    # Generate TOC with links
    for script in scripts:
        module_name = script.replace('/', '.').rstrip('.py')
        combined_doc.write(f'<li><a href="#{module_name}">{module_name}</a></li>')

    combined_doc.write('</ul>')

    # Add documentation content with section IDs for linking
    for script in scripts:
        module_name = script.replace('/', '.').rstrip('.py')
        html_filename = os.path.join(temp_doc_dir, f"{module_name}.html")
        if os.path.exists(html_filename):
            with open(html_filename, 'r', encoding='utf-8') as f:
                # Extract the body content to avoid duplicate <html> tags
                content = f.read().split('<body>')[1].split('</body>')[0]
                combined_doc.write(f'<h2 id="{module_name}">{module_name}</h2>')
                combined_doc.write(content)
                combined_doc.write('<hr>')  # Add a separator between sections
            os.remove(html_filename)  # Clean up the temporary file

    combined_doc.write('</body></html>')

# Remove the temporary docs directory
os.rmdir(temp_doc_dir)

print(f"Combined documentation with TOC has been created: {combined_doc_path}")
