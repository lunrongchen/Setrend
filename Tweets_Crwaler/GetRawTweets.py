import oauth2
import time
import urllib2
import json
from pymongo import MongoClient


class GetTweets:

    """
    This class is to get the tweets from the twitter
       """

    def __init__(self, since_id, max_id, keywords, keywords_id, db, collection):
        self._max_id = max_id
        self._keywords = keywords
        self._keywords_id = keywords_id
        self._since_id = since_id
        self._collection = collection
        self.start_id = 0

    def getTweets(self):

        collection = self._collection
        url1 = "https://api.twitter.com/1.1/search/tweets.json"
        params = {
            "oauth_version": "1.0",
            "oauth_nonce": oauth2.generate_nonce(),
            "oauth_timestamp": int(time.time())
        }

        consumer = oauth2.Consumer(key="YOU_CONSUMER_KEY_FROM_TWITTER_API",
                                   secret="YOU_CONSUMER_SECRET_FROM_TWITTER_API")
        token = oauth2.Token(key="YOU_TOKEN_KEY_FROM_TWITTER_API",
                             secret="YOU_TOKEN_SECRET_FROM_TWITTER_API")

        params["oauth_consumer_key"] = consumer.key
        params["oauth_token"] = token.key

        # loop
        for i in range(100000):
            print "sin_id:  ", self._since_id
            print "max_id:  ", self._max_id
            url = url1
            params["q"] = str(self._keywords)     
            params["count"] = 100
            params["geocode"] = ""
            params["lang"] = "en"
            params["locale"] = ""
            params["result_type"] = ""
            params["until"] = ""
            params["since_id"] = str(self._since_id)
            params["max_id"] = str(self._max_id)

            req = oauth2.Request(method="GET", url=url, parameters=params)
            signature_method = oauth2.SignatureMethod_HMAC_SHA1()
            req.sign_request(signature_method, consumer, token)
            headers = req.to_header()
            url = req.to_url()
            response = urllib2.Request(url)

            data = json.load(urllib2.urlopen(response))
            if data["statuses"] == []:
                print "end of data"
                break
            else:
                # current_id is what the spider crawl
                self._max_id = int(data["statuses"][-1]["id"]) - 1
                print "The", self._keywords_id, "th keyword ", self._keywords, "  are crawl in  ", self._max_id, "  ###  ", i, "  runing now   "
                if self.start_id < self._max_id:
                    self.start_id = self._max_id
            collection.insert(data["statuses"])  # insert data into files
            time.sleep(5)
