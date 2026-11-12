import unittest
from unittest.mock import Mock

import requests
from movie_api import fetch_movie, movie_fields, MovieAPIError


class MovieTests(unittest.TestCase):
    def session(self, payload=None, status=200, error=None):
        response = Mock(status_code=status)
        response.json.return_value = payload
        session = Mock()
        if error:
            session.get.side_effect = error
        else:
            session.get.return_value = response
        return session

    def test_existing_movie(self):
        payload = {'Response': 'True', 'Title': 'Batman', 'Year': '1989'}
        session = self.session(payload)
        result = fetch_movie(' Batman ', 'secret', session=session)
        self.assertEqual(result['Title'], 'Batman')
        self.assertEqual(movie_fields(result)['Актёры'], 'Нет данных')
        self.assertEqual(session.get.call_args.kwargs['timeout'], 20)

    def test_nonexistent_movie(self):
        with self.assertRaisesRegex(MovieAPIError, 'Фильм не найден'):
            fetch_movie('no_such_movie', 'secret', session=self.session({'Response':'False','Error':'Movie not found!'}))

    def test_wrong_key_http(self):
        with self.assertRaisesRegex(MovieAPIError, 'доступ'):
            fetch_movie('Batman', 'invalid', session=self.session(status=401))

    def test_wrong_key_json(self):
        with self.assertRaisesRegex(MovieAPIError, 'ключ'):
            fetch_movie('Batman', 'invalid', session=self.session({'Response':'False','Error':'Invalid API key!'}))

    def test_timeout_does_not_reveal_key(self):
        with self.assertRaises(MovieAPIError) as caught:
            fetch_movie('Batman', 'secret', session=self.session(error=requests.Timeout('url?apikey=secret')))
        self.assertNotIn('secret', str(caught.exception))

    def test_bad_json(self):
        session=self.session()
        session.get.return_value.json.side_effect=ValueError('bad json')
        with self.assertRaisesRegex(MovieAPIError, 'JSON'):
            fetch_movie('Batman', 'secret', session=session)

    def test_empty_inputs_do_not_send_requests(self):
        session=self.session()
        for title, key in [('', 'secret'), ('Batman', '')]:
            with self.assertRaises(MovieAPIError):
                fetch_movie(title, key, session=session)
        session.get.assert_not_called()

    def test_server_error(self):
        with self.assertRaisesRegex(MovieAPIError, 'HTTP 500'):
            fetch_movie('Batman', 'secret', session=self.session(status=500))


if __name__ == '__main__':
    unittest.main()
