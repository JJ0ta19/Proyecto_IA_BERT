# -*- coding: utf-8 -*-
"""Genera docs/documentacion-completa.html a partir de los .md de docs/"""
import os
import re
from datetime import datetime
import markdown

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS = [
    ("01-instalacion.md", "1. Instalación y Configuración"),
    ("02-arquitectura-tecnologias.md", "2. Arquitectura y Tecnologías"),
    ("03-red-neuronal-bert.md", "3. Red Neuronal BERT"),
    ("04-datos-entrenamiento.md", "4. Datos de Entrenamiento"),
    ("05-metricas-graficas.md", "5. Métricas y Gráficas"),
]

md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "nl2br"])
sections = []
toc_items = []

for fname, title in DOCS:
    path = os.path.join(BASE, fname)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # Reescribir enlaces internos entre .md para que apunten a los anclajes del propio documento
    text = re.sub(r"\]\((\d{2}-[a-z\-]+\.md)(#[^)]*)?\)",
                  lambda m: f"](#sec-{m.group(1)[:2]}{m.group(2) or ''})", text)
    html = md.convert(text)
    md.reset()
    sid = f"sec-{fname[:2]}"
    sections.append(f'<section id="{sid}">{html}</section>')
    toc_items.append(f'<li><a href="#{sid}">{title}</a></li>')

today = datetime.now().strftime("%Y-%m-%d")

html_doc = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Proyecto IA BERT - Documentación Completa</title>
<style>
  :root {{ --primary: #2563eb; --dark: #1e293b; --light: #f1f5f9; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; color: var(--dark); line-height: 1.65; }}
  header {{ background: linear-gradient(135deg, #1e3a8a, #2563eb); color: #fff; padding: 48px 24px; text-align: center; }}
  header h1 {{ margin: 0 0 8px; font-size: 2em; }}
  header p {{ margin: 0; opacity: .9; }}
  .meta {{ font-size: .85em; opacity: .75; margin-top: 10px; }}
  nav {{ background: var(--light); padding: 20px 32px; border-bottom: 3px solid var(--primary); }}
  nav strong {{ display: block; margin-bottom: 10px; }}
  nav ol {{ margin: 0; padding-left: 22px; columns: 2; }}
  main {{ max-width: 960px; margin: 0 auto; padding: 16px 32px 64px; }}
  section {{ border-bottom: 1px solid #e2e8f0; padding-top: 12px; }}
  h1 {{ color: var(--primary); border-bottom: 2px solid var(--primary); padding-bottom: 6px; }}
  h2 {{ color: #1d4ed8; margin-top: 1.8em; }}
  code {{ background: var(--light); padding: 2px 6px; border-radius: 4px; font-size: .92em; }}
  pre {{ background: #0f172a; color: #e2e8f0; padding: 14px 18px; border-radius: 8px; overflow-x: auto; }}
  pre code {{ background: none; color: inherit; }}
  table {{ border-collapse: collapse; width: 100%; margin: 14px 0; font-size: .95em; }}
  th {{ background: var(--primary); color: #fff; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 8px 12px; text-align: left; }}
  tr:nth-child(even) td {{ background: var(--light); }}
  blockquote {{ border-left: 4px solid var(--primary); margin: 14px 0; padding: 8px 18px; background: var(--light); }}
  footer {{ background: var(--dark); color: #94a3b8; text-align: center; padding: 18px; font-size: .85em; }}
</style>
</head>
<body>
<header>
  <h1>&#129302; Proyecto IA BERT</h1>
  <p>Sistema de Clasificación de Currículums con Red Neuronal BERT</p>
  <div class="meta">Django · PyTorch · Transformers · spaCy · Tesseract OCR &nbsp;|&nbsp; v1.0 &nbsp;|&nbsp; {today}</div>
</header>
<nav>
  <strong>Contenido</strong>
  <ol>
    {''.join(toc_items)}
  </ol>
</nav>
<main>
{''.join(sections)}
</main>
<footer>Proyecto IA BERT — Documentación técnica completa · Generada el {today}</footer>
</body>
</html>"""

out = os.path.join(BASE, "documentacion-completa.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html_doc)
print(f"OK -> {out} ({os.path.getsize(out)//1024} KB)")
