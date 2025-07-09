from nicegui import ui
import numpy as np
from matplotlib import pyplot as plt

ui.markdown('# Dashboard')

with ui.button_group().props('rounded glossy'):
    ui.button('RNN', color='blue').props('push')
    ui.button('CNN', color='green').props('push text-color=black')
    ui.button('SNN', color='purple').props('push text-color=white')
    ui.button('FSNN', color='pink').props('push text-color=black')

with ui.pyplot(figsize=(3, 2)):
    x = np.linspace(0.0, 5.0)
    y = np.cos(2 * np.pi * x) * np.exp(-x)
    plt.plot(x, y, '-')

radio1 = ui.radio(['Rewards', 'Runtime'], value=1).props('inline')

ui.markdown('### Best Reward: Worst Reward: Average Reward:')

ui.run()

