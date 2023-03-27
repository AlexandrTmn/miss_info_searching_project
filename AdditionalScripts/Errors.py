from AdditionalScripts import Auth


class CheckErrors:
    def check_id(tweet_id):
        # try:
        #     tweet = Auth.api.lookup_statuses(tweet_id)
        # except (tweepy.errors.NotFound, tweepy.errors.Forbidden) as e:
        #     return False

        try:
            Auth.client.get_retweeters(tweet_id).meta['result_count']
        except KeyError:
            retweeters = 0
        else:
            retweeters = Auth.client.get_retweeters(tweet_id).meta['result_count']
        return retweeters
