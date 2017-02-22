from zipline.api import (
    history,
    order_target,
    record,
    symbol,
)


def initialize(context):
    context.security = symbol('SPY')


def handle_data(context, data):
	print data
	MA1 = data[context.security].mavg(50)
	MA2 = data[context.security].mavg(200)

	current_Price = data[context.security].price
	current_positions = context.portfolio[symbol('SPY')].amount
	cash = context.portfolio.cash
