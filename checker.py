# topics word distribution checker
import sys
import math

if(len(sys.argv) != 3):
	print "python checker.py __vocabfile__ __topicfile__"
	sys.exit()

vocabfile = sys.argv[1]
topicfile = sys.argv[2]

vocabs=[""]
with open(vocabfile) as f:
	for line in f:
		vocabs.append(line.strip())

with open(topicfile) as f:
	for line in f:
		words=line.strip().split(", ")
		for word in words:
			# if(word=="\n"): 
				# continue
			w_p = word.split(":")
			v = vocabs[int(w_p[0])]
			l = int(math.ceil(float(len(v) + 10) / 20) * 20)
			s = v + ":" + w_p[1] + ","
			sys.stdout.write(s.rjust(l))
		print "\n"
		sys.stdout.flush()

