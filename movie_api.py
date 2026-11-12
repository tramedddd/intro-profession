"""OMDb: обработка HTTP, ошибок сервиса и неполных ответов."""
import requests


class MovieAPIError(RuntimeError):
    """Сообщение, которое можно безопасно показать пользователю."""


def fetch_movie(title, api_key, *, session=requests):
    title, api_key = title.strip(), api_key.strip()
    if not title:
        raise MovieAPIError('Введите непустое название фильма.')
    if not api_key:
        raise MovieAPIError('Не задан ключ OMDb API.')
    try:
        response = session.get('https://www.omdbapi.com/',
                               params={'t': title, 'apikey': api_key}, timeout=20)
    except requests.Timeout:
        raise MovieAPIError('OMDb не ответил вовремя. Повторите позже.') from None
    except requests.RequestException:
        raise MovieAPIError('Не удалось подключиться к OMDb.') from None
    if response.status_code in (401, 403):
        raise MovieAPIError('OMDb отклонил доступ: проверьте API-ключ.')
    if response.status_code != 200:
        raise MovieAPIError(f'OMDb вернул HTTP {response.status_code}.')
    try:
        data = response.json()
    except ValueError:
        raise MovieAPIError('OMDb вернул некорректный JSON.') from None
    if not isinstance(data, dict):
        raise MovieAPIError('OMDb вернул неожиданный формат ответа.')
    if data.get('Response') == 'False':
        # Не отражаем произвольный ответ сервиса: он может содержать входные данные.
        reason = str(data.get('Error', '')).lower()
        if 'key' in reason or 'limit' in reason:
            raise MovieAPIError('Проверьте ключ OMDb и доступную квоту запросов.')
        raise MovieAPIError('Фильм не найден. Уточните название.')
    if data.get('Response') != 'True' or not data.get('Title'):
        raise MovieAPIError('В ответе OMDb отсутствуют данные фильма.')
    return data


def movie_fields(data):
    fields = {'Название': 'Title', 'Год': 'Year', 'Жанр': 'Genre',
              'Режиссёр': 'Director', 'Рейтинг IMDb': 'imdbRating', 'Актёры': 'Actors'}
    return {label: data.get(key) or 'Нет данных' for label, key in fields.items()}
