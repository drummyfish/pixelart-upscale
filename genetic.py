from PIL import Image
import random
import math

NS = 9              # neighbour size
N_MAX = NS * NS - 1 # maximum pixel number in the neighbourhood
MAX_CONDITIONS = 100

class NeighbourhoodCondition(object):

  def __init__(self):
    self.number_of_operands = 2
    self.operands = []    # list of either conditions (for logical conditions) or numbers (pixels or parameters, for terminal conditions) 
   
  def to_python_code(self):
    return "???"

  def remap_references(self, new_remap):
    pass

  def references_condition(self, index):
    return False

class ConditionReference(NeighbourhoodCondition):     # reference to a separate condition in condition list
  def __init__(self, condition_index):
    super(ConditionReference,self).__init__()
    self.number_of_operands = 1
    self.operands = [condition_index]

  def remap_references(self, new_remap):
    self.operands[0] = new_remap[self.operands[0]]

  def references_condition(self, index):
    return index == self.operands[0]

  def to_python_code(self):
    if self.operands[0] < 0:     # non-existing
      return "True"

    return "c" + str(self.operands[0])

class ConditionLogical(NeighbourhoodCondition):

   def __init__(self):
     super(ConditionLogical,self).__init__()

   def remap_references(self, new_remap):
     for child in self.operands:
       child.remap_references(new_remap)

   def references_condition(self, index):
     for child in self.operands:
       if child.references_condition(index):
         return True

     return False

class ConditionAnd(ConditionLogical):

  def __init__(self, condition_a, condition_b):
    super(ConditionAnd,self).__init__()
    self.operands = [condition_a, condition_b]

  def to_python_code(self):
    return "(" + self.operands[0].to_python_code() + ") and (" + self.operands[1].to_python_code() + ")"

class ConditionOr(ConditionLogical):

  def __init__(self, condition_a, condition_b):
    super(ConditionOr,self).__init__()
    self.operands = [condition_a, condition_b]

  def to_python_code(self):
    return "(" + self.operands[0].to_python_code() + ") or (" + self.operands[1].to_python_code() + ")"

class ConditionXor(ConditionLogical):

  def __init__(self, condition_a, condition_b):
    super(ConditionXor,self).__init__()
    self.operands = [condition_a, condition_b]

  def to_python_code(self):
    return "(" + self.operands[0].to_python_code() + ") xor (" + self.operands[1].to_python_code() + ")"

class ConditionNot(ConditionLogical):

  def __init__(self, condition):
    super(ConditionNot,self).__init__()
    self.number_of_operands = 1
    self.operands = [condition]

  def to_python_code(self):
    return "not (" + self.operands[0].to_python_code() + ")"

class ConditionPixelsAreEqual(NeighbourhoodCondition):

  def __init__(self, pixel_a, pixel_b):
    super(ConditionPixelsAreEqual,self).__init__()
    self.operands = [pixel_a, pixel_b]

  def to_python_code(self):
    return "equal(p[" + str(self.operands[0]) + "],p["  + str(self.operands[1]) + "])"

class ConditionPixelIsBrighter(NeighbourhoodCondition):

  def __init__(self, pixel_a, pixel_b):
    super(ConditionPixelIsBrighter,self).__init__()
    self.operands = [pixel_a, pixel_b]

  def to_python_code(self):
    return "brighter(p[" + str(self.operands[0]) + "],p["  + str(self.operands[1]) + "])"

# Algorithm that filters the image. For neighbourhood
# of size NS * NS with pixels numbered as
# 
#   0  1 2 3 ... (NS - 1)
#   NS (NS + 1) ...
#   .
#   .
#   .
#   ...          NS^2 - 1
# 
# outputs 4 pixels (upscales the image 2x) numbered as
#
# 0 1
# 2 3
#
# Works as follows: There is an ordered set of conditions 
# on the pixel neighbourhood, the conditions are:
#
# p0, p1, p2, ... pn
#
# Each condition may use previous conditions, but not
# the following ones.
#
# Then for each output pixel there is a case/switch-like
# expression on the condition to output one of the pixels
# of the neighbourhood.

class UpscaleAlgorithm:

  def __init__(self):

    self.conditions = []
    self.pixel0_output = []     # nonempty list of tuples in format (condition_index, pixel number), last condition is ignored
    self.pixel1_output = []
    self.pixel2_output = []
    self.pixel3_output = []

  def to_python_code(self):
    result = ""

    i = 0

    for condition in self.conditions:
      result += "c" + str(i) + " = " + condition.to_python_code() + "\n"
      i += 1

    r = 0

    for if_statement in (self.pixel0_output,self.pixel1_output,self.pixel2_output,self.pixel3_output):
      result += "\n"

      for i in range(len(if_statement)):

        if len(if_statement) != 1:
          if i == len(if_statement) - 1:
            result += "else:\n"
          else:
            if i == 0:
              result += "if c" + str(if_statement[i][0]) + ":\n"
            else:
              result += "elif c" + str(if_statement[i][0]) + ":\n"
          
          result += "  "

        result += "r" + str(r) + " = p[" + str(if_statement[i][1]) + "] \n"

      r += 1

    return result

  def delete_condition(self, index):
    index_remap = [i if (i < index) else (i - 1) for i in range(len(self.conditions))]
    index_remap[index] = -1

    del self.conditions[index] 

    # remap reference conditions

    for condition in self.conditions:
      condition.remap_references(index_remap)

    # remap the switches

    def fix_output(output):
      return map(lambda item: item if item[0] <= index else (item[0] - 1,item[1]),output)

    self.pixel0_output = fix_output(self.pixel0_output)
    self.pixel1_output = fix_output(self.pixel1_output)
    self.pixel2_output = fix_output(self.pixel2_output)
    self.pixel3_output = fix_output(self.pixel3_output)

  def normalize(self):     # cleans the algorithm (drops unused conditions etc.)
    used = [False for c in self.conditions]

    for i in range(len(self.conditions)):
      for j in range(i + 1,len(self.conditions)):
        if self.conditions[j].references_condition(i):
          used[i] = True
          break

    for output in [self.pixel0_output,self.pixel1_output,self.pixel2_output,self.pixel3_output]:
      for item in output[:-1]:  # last item's condition is not used
        used[item[0]] = True

    print(used)

    for i in reversed(range(len(self.conditions))):
      if not used[i]:
        self.delete_condition(i)

class RandomGenerator(object):
 
  def __init__(self, seed_number=-1):
    if seed_number >= 0:
      random.seed(seed_number)

  def generateRandomCondition(self, condition_index, max_depth=2, generate_reference=False):

    random_number = random.randint(0,5 if (condition_index == 0 or not generate_reference) else 6)

    if random_number in (0,1) or max_depth == 0:
      pixel1 = random.randint(0,N_MAX)
      pixel2 = pixel1

      while pixel2 == pixel1:                 # make sure to have different pixels
        pixel2 = random.randint(0,N_MAX)

      if random_number == 0:
        return ConditionPixelsAreEqual( pixel1, pixel2 )
      else:
        return ConditionPixelIsBrighter( pixel1, pixel2 )
    elif random_number in (2,3,4,5):

      condition1 = self.generateRandomCondition(condition_index, max_depth - 1,True)

      if random_number != 5:
        condition2 = self.generateRandomCondition(condition_index, max_depth - 1,True)

      if random_number == 2:
        return ConditionAnd( condition1, condition2)
      elif random_number == 3:
        return ConditionOr( condition1, condition2)
      elif random_number == 4:
        return ConditionXor( condition1, condition2)
      else:
        return ConditionNot( condition1 )
    else:
      return ConditionReference( random.randint(0,condition_index - 1) )

  def generateRandomAlgorithm(self):
    result = UpscaleAlgorithm()

    def random_switch_statement(alg):
      res = []

      indices = range(len(alg.conditions))
      random.shuffle(indices)

      for i in indices:
        res.append( (i,random.randint(0,N_MAX)) )

        if random.randint(0,2) == 0:
          break

      return res

    for i in range(MAX_CONDITIONS):
      result.conditions.append(self.generateRandomCondition(i))

      if random.randint(0,5) == 0:
        break

    result.pixel0_output = random_switch_statement(result)
    result.pixel1_output = random_switch_statement(result)
    result.pixel2_output = random_switch_statement(result)
    result.pixel3_output = random_switch_statement(result)

    return result

r = RandomGenerator(31)

a1 = r.generateRandomAlgorithm()

print(a1.to_python_code())

a1.normalize()

print(a1.to_python_code())
