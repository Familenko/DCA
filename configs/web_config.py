import datetime


TODAY = datetime.datetime.now().strftime("%Y-%m-%d")
YEAR_AGO = (datetime.datetime.now() - datetime.timedelta(days=365 * 3)).strftime("%Y-%m-%d")

ASSETS = {
    "Crypto": [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 
        'XRP-USD', 'SOL-USD', 'DOT-USD', 
        'ADA-USD', 'LINK-USD', 'AVAX-USD'
        ],
    "Stocks": ['SPY'],
}

MENU_ITEMS = {
    'Get Help': 'https://github.com/Familenko/DCA/blob/main/README.md',
    'Report a bug': "mailto:leshafamilenko@gmail.com",
    'About': "This app allows you to backtest a DCA strategy with take-profit and cooldown features. Configure the settings in the sidebar and click 'Run backtest' to see the results"
    }

INSTRUCTIONS = """
## Що робить застосунок

Застосунок симулює DCA-стратегію (регулярні покупки активу) з алгоритмічним take-profit продажем і періодом cooldown.
Після запуску завантажується історія цін, виконується покроковий бектест, показуються графік, події продажів та фінальна таблиця.

## Як користуватись

1. Оберіть тип активу та тикер.
2. Задайте період тесту (Start date / End date).
3. Налаштуйте параметри стратегії в sidebar.
4. Натисніть **Run backtest**.
5. Перегляньте графік, take-profit події та результат у таблиці.

## Параметри (sidebar)

- **Asset Type**: категорія активів (Crypto або Stocks).
- **Asset**: конкретний тикер (наприклад BTC-USD або SPY).
- **Start date / End date**: період історичних даних для симуляції.
- **Buy amount**: сума однієї покупки в базовій валюті портфеля.
- **Frequency**: частота запланованих покупок.
    - `W-MON` - щотижня в понеділок.
    - `WOM-1MON` - перший понеділок місяця.
    - `D` - щодня.
- **Minimum profit**: поріг продажу відносно собівартості позиції.
    - поточна вартість позиції >= **minimum_profit** * базова вартість позиції.
- **Minimum loss**: поріг докупки відносно собівартості позиції.
    - поточна вартість позиції <= **minimum_loss** * базова вартість позиції.
- **Enable selling**: увімкнути продаж.
- **Manual sell fraction**: увімкнути фіксовану частку продажу.
    - **Sell fraction**: частка позиції для продажу (0.0..1.0), якщо ввімкнено fixed режим.
- **Cooldown days**: кількість днів блокування наступного take-profit після фактичного продажу.
- **Cooldown wait**: короткий cooldown, якщо умова оцінки продажу спрацювала, але умова продажу не виконана.
- **Fee %**: відсоток комісії за транзакцію.
- **Max invest years**: продаж при накопиченні інвестицій на певну кількість років.

## Як працює логіка бектесту

- Дані цін завантажуються через Yahoo Finance, використовується колонка `Close`.
- На датах згідно Frequency виконується перевірка покупки.
- Продаж проходить через модулі рішень
- Якщо обрано Manual sell fraction, то продаж відбувається на вказану частку позиції
- Умова take-profit перевіряється щодня через Minimum profit, але продаж може бути заблокований через cooldown після попереднього продажу.
- Фіксуються:
    - `Realized_profit` - прибуток від конкретного продажу,
    - `Returns` - грошовий притік від продажу,
    - `Trigger_msg` - джерело/пояснення рішення продажу.

## Важливо
Дефолтні параметри є найбільш оптимальними згідно з проведеними тестами, але ви можете експериментувати з ними для кращого розуміння стратегії. Застосунок призначений для освітніх цілей і не є фінансовою порадою.
"""