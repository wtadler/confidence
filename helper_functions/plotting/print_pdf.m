function print_pdf(filename)

set(gcf, 'paperunits', 'points')
pos = get(gcf, 'position');
set(gcf, 'papersize', pos(3:4))
set(gcf, 'paperpositionmode','auto')
print(gcf, '-dpdf', filename)