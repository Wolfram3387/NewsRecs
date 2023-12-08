from keyboards.inline import get_markup
from loader import dp, bot

from utils.db_api.api import get_all_users
from utils.neural_networks.get_distance import get_distance
from utils.parser import get_new_news


async def check_news():
    new_df = get_new_news()
    if new_df is None:
        return
    all_users = get_all_users()
    for news in new_df.itertuples():
        news_keywords = news.Keywords
        for user in all_users:
            user_keywords = user[1]
            if get_distance(user_keywords, news_keywords) < 0.87:
                user_id = user[0]
                await send_news(user_id, news)


async def send_news(user_id, news):
    markup = get_markup(news.ID)
    await bot.send_message(chat_id=user_id, text=news.Link, reply_markup=markup)
