import click
import pickle
import neptune
from reports.pdf_creator import create_report
from methods.q_learning import q_learning
from methods.monte_carlo import monte_carlo
from methods.n_steps import n_steps
from methods.dqn import dqn_learning
from methods.ddqn import ddqn_learning


neptune.init(project_qualified_name='dmitriygorbatovskii/sandbox', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMjI0MjA1YWUtMmFmOC00OTlhLTkzNGItNTBmYjNlNzkwMmEwIn0=')
neptune.create_experiment()

@click.command()
@click.option('--weather', default=0, help='set weather')
@click.option('--time', default=12, help='set time')
@click.option('--alpha', default=0.1, help='step')
@click.option('--decay', default=500, help='epsilon decay')
@click.option('--gma', default=1, help='discount factor')
@click.option('--epochs', default=1, help='epochs')
@click.option('--steps', default=1000000, help='steps')
@click.option('--agent', default='Lincoln2017MKZ (Apollo 5.0)', help='Jaguar2015XE (Apollo 3.0),\n'
                                                                     'Lexus2016RXHybrid (Autoware),\n'
                                                                     'Lincoln2017MKZ (Apollo 5.0)')
@click.option('--train', default=True, help='True - train new model,'
                                            'False - demonstration of the trained model')
@click.option('--type', default='ddqn', help='q_learning, monte_carlo, n_steps, dqn, ddqn')
def main(weather, time, alpha, decay, gma, epochs, steps, agent, train, type):
    data_name = 'data/data_{}.pickle'.format(type)

    if type == 'q_learning':
        data = q_learning(alpha, gma, epochs, train)
    elif type == 'monte_carlo':
        data = monte_carlo(alpha, gma, epochs, train)
    elif type == 'n_steps':
        data = n_steps()
    elif type == 'dqn':
        data = dqn_learning(decay, gma, steps, train)
    elif type == 'ddqn':
        data = ddqn_learning(decay=100, gma=0.99, steps=10**9, train=True)


    if data:
        with open(data_name, 'wb') as f:
            pickle.dump(data, f)

    create_report(alpha, decay, gma, epochs, steps, agent, learning_type=type)


if __name__ == "__main__":
    main()