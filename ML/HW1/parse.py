import csv
from utilities import most_frequent


def missing_vals(data):

  common_value_by_key = {}
  for key in data[0].keys():
    common_values = []
    for row in data:
      val = row[key]
      common_values.append(val)
    common_value_by_key[key] = most_frequent(common_values)

  for example in data:
    for key in example.keys():
      if example[key] == '?':
        example[key] = common_value_by_key[key]

  return data


def parse(filename):
  '''
  takes a filename and returns attribute information and all the data in array of dictionaries
  '''
  # initialize variables

  out = []  
  csvfile = open(filename,'r')
  fileToRead = csv.reader(csvfile)

  headers = next(fileToRead)

  # iterate through rows of actual data
  for row in fileToRead:
    out.append(dict(zip(headers, row)))

  return missing_vals(out)

x = parse('house_votes_84.data')
