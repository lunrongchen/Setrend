# 1.0*e15
import time
import MongoCrawl

def RunInterval():
    since_id = 710000000000000000
    max_id = 719000000000000000
    i = 1
    while i:
        print "This is Dow Jones ", i, "th loop starts"
        MongoCrawl.CrawlTweetsToMongoDB(since_id, max_id)
        print "This is Dow Jones ", i, "th loop ends"
        print " "
        print "#########################################"
        print " "
        since_id = max_id  # since_id is what the old max_id plus one
        max_id = max_id + 1000000000000000
        # The interval are 80 hours
        for ii in range(50000):
            print "This is Dow Jones ", i, "th interval running"
            print " "
            time.sleep(5)
            print "Interval remainning time is", (57600 - ii) / 720, " hours left"
            print " "
        i = i + 1

RunInterval()
