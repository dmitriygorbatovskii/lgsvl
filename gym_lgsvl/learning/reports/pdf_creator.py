from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus.tables import Table


def create_report(alpha=0, decay=0, gma=0, epochs=0, steps=0, agent=0, learning_type='q_learning'):
    metric = 'reward/epochs'
    conditions = 'rain = 0, fog = 0, time = 12:00'
    time = '12:34:56'
    img = {
        'q_learning': 'reports/ql.png',
        'monte_carlo': 'reports/mc.png',
        'n_steps': 'reports/ns.png',
        'dqn': 'reports/dqn.png',
        'ddqn': 'reports/ddqn.png',
    }

    doc = SimpleDocTemplate("reports/report.pdf", pagesize=letter,
                            rightMargin=20, leftMargin=30,
                            topMargin=20, bottomMargin=18)

    Story = []
    styles = getSampleStyleSheet()

    data = [
            ('alpha', alpha),
            ('decay', decay),
            ('gma', gma),
            ('epochs', epochs),
            ('steps', steps),
            ('agent: ', agent),
            ('learning type:', learning_type),
            ('conditions:', conditions),
            ('lead time: ', time),
            ]
    table = Table(data, colWidths=270, rowHeights=20)
    Story.append(table)

    Story.append(Spacer(1, 12))
    ptext = '<font size="12">metrics: %s </font>' % metric
    Story.append(Paragraph(ptext, styles["Normal"]))
    Story.append(Spacer(1, 12))

    im = str([img[i] for i in img if i == learning_type][0])
    im = Image(im, 400, 300)
    Story.append(im)

    doc.build(Story)






