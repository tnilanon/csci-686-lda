# topics' word distributions checker

import sys
import math

screen_width = 80
column_width = 20

if len(sys.argv) < 3 or len(sys.argv) > 4:
	print "python checker.py __vocabfile__ __topicfile__ [nice]"
	sys.exit()

vocabfile = sys.argv[1]
topicfile = sys.argv[2]

if len(sys.argv) == 4 and sys.argv[3] == "nice":
	nice = True
else:
	nice = False

vocabs = [""]
with open(vocabfile) as f:
	for line in f:
		vocabs.append(line.strip())

with open(topicfile) as f:
	for line in f:
		word_prob_pairs = line.strip().split(", ")
		num_remaining_characters = screen_width
		for word_prob_pair in word_prob_pairs:
			temp = word_prob_pair.split(":")
			word = vocabs[int(temp[0])]
			to_print = word + ":" + temp[1] + ","
			if nice:
				num_columns = int(math.ceil(float(len(to_print)) / column_width))
				num_characters = num_columns * column_width
				if num_remaining_characters < num_characters:
					t = num_characters
					num_characters += num_remaining_characters
					num_remaining_characters = screen_width - t
				else:
					num_remaining_characters -= num_characters
				sys.stdout.write(to_print.rjust(num_characters))
			else:
				sys.stdout.write(to_print)
		if nice:
			print "  ------  \n"
		else:
			print ""
		sys.stdout.flush()

