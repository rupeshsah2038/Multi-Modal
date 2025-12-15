#!/bin/bash
# Quick script to regenerate all research figures

echo "Regenerating all research article figures..."
python tools/plot_research_figures.py --output-dir figures/research_article --plots all

echo ""
echo "✓ Figures generated in: figures/research_article/"
echo ""
echo "Generated files:"
ls -lh figures/research_article/*.pdf | wc -l | xargs echo "  - PDF files:"
ls -lh figures/research_article/*.png | wc -l | xargs echo "  - PNG files:"
echo ""
echo "To view figures:"
echo "  cd figures/research_article && xdg-open *.png"
