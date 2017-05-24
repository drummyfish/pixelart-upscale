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

  def correct_references(self, index):   # makes sure references are only made to previous conditions
    return

  def simplified(self):
    return self

  def alwaysTrue(self):
    return False

  def alwaysFalse(self):
    return False

  def simplify_children(self):
    for i in range(len(self.operands)):
      self.operands[i] = self.operands[i].simplified()

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

  def correct_references(self, index):
    if self.operands[0] >= index:
      self.operands[0] = -1

  def alwaysTrue(self):
    return self.operands[0] == -1

  def simplified(self):
    if self.alwaysTrue():
      return ConditionTrue()

    return self

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

   def correct_references(self, index):
     for child in self.operands:
       child.correct_references(index)

class ConditionAnd(ConditionLogical):

  def __init__(self, condition_a, condition_b):
    super(ConditionAnd,self).__init__()
    self.operands = [condition_a, condition_b]

  def to_python_code(self):
    return "(" + self.operands[0].to_python_code() + ") and (" + self.operands[1].to_python_code() + ")"

  def alwaysFalse(self):
    return self.operands[0].alwaysFalse() or self.operands[1].alwaysFalse()

  def simplified(self):
    if self.alwaysFalse():
      return ConditionFalse()
    elif self.operands[0].alwaysTrue():
      return self.operands[1]
    elif self.operands[1].alwaysTrue():
      return self.operands[0]

    self.simplify_children()

    return self

class ConditionOr(ConditionLogical):

  def __init__(self, condition_a, condition_b):
    super(ConditionOr,self).__init__()
    self.operands = [condition_a, condition_b]

  def to_python_code(self):
    return "(" + self.operands[0].to_python_code() + ") or (" + self.operands[1].to_python_code() + ")"   

  def alwaysTrue(self):
    return self.operands[0].alwaysTrue() or self.operands[1].alwaysTrue()

  def simplified(self):
    if self.alwaysTrue():
      return ConditionTrue()
    elif self.operands[0].alwaysFalse():
      return self.operands[1]
    elif self.operands[1].alwaysFalse():
      return self.operands[0]     

    self.simplify_children()

    return self
 
class ConditionXor(ConditionLogical):

  def __init__(self, condition_a, condition_b):
    super(ConditionXor,self).__init__()
    self.operands = [condition_a, condition_b]

  def to_python_code(self):
    return "(" + self.operands[0].to_python_code() + ") xor (" + self.operands[1].to_python_code() + ")"

  def simplified(self):
    if self.operands[0].alwaysFalse():
      return ConditionNot(self.operands[1])
    elif self.operands[0].alwaysTrue():
      return self.operands[1]   
    if self.operands[1].alwaysFalse():
      return ConditionNot(self.operands[0])
    elif self.operands[1].alwaysTrue():
      return self.operands[0]   

    self.simplify_children()

    return self

class ConditionNot(ConditionLogical):

  def __init__(self, condition):
    super(ConditionNot,self).__init__()
    self.number_of_operands = 1
    self.operands = [condition]

  def to_python_code(self):
    return "not (" + self.operands[0].to_python_code() + ")"

  def alwaysFalse(self):
    return self.operands[0].alwaysTrue()

  def simplified(self):
    if self.operands[0].alwaysTrue():
      return ConditionFalse()
    elif self.operands[0].alwaysFalse():
      return ConditionTrue()

    return self

class ConditionTrue(ConditionLogical):

  def __init__(self):
    super(ConditionTrue,self).__init__()
    self.number_of_operands = 0

  def to_python_code(self):
    return "True"

  def alwaysTrue(self):
    return True

class ConditionFalse(ConditionLogical):

  def __init__(self):
    super(ConditionFalse,self).__init__()
    self.number_of_operands = 0

  def to_python_code(self):
    return "False"

  def alwaysFalse(self):
    return True

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
    # correct outputs

    def correct_output(output):
      return filter(lambda item: item[0] < len(self.conditions),output[:-1]) + [output[-1]]

    self.pixel0_output = correct_output(self.pixel0_output)
    self.pixel1_output = correct_output(self.pixel1_output)
    self.pixel2_output = correct_output(self.pixel2_output)
    self.pixel3_output = correct_output(self.pixel3_output)

    # correct condition references

    for i in range(len(self.conditions)):
      self.conditions[i].correct_references(i)

    # remove unused conditions:

    used = [False for c in self.conditions]

    for i in range(len(self.conditions)):
      for j in range(i + 1,len(self.conditions)):
        if self.conditions[j].references_condition(i):
          used[i] = True
          break

    for output in [self.pixel0_output,self.pixel1_output,self.pixel2_output,self.pixel3_output]:
      for item in output[:-1]:  # last item's condition is not used
        used[item[0]] = True

    for i in reversed(range(len(self.conditions))):
      if not used[i]:
        self.delete_condition(i)

    for i in range(len(self.conditions)):
      self.conditions[i] = self.conditions[i].simplified()

class RandomGenerator(object):
 
  def __init__(self, seed_number=-1):
    if seed_number >= 0:
      random.seed(seed_number)

  def generate_random_condition(self, condition_index, max_depth=2, generate_reference=False):

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

      condition1 = self.generate_random_condition(condition_index, max_depth - 1,True)

      if random_number != 5:
        condition2 = self.generate_random_condition(condition_index, max_depth - 1,True)

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

  def generate_random_algorithm(self):
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
      result.conditions.append(self.generate_random_condition(i))

      if random.randint(0,5) == 0:
        break

    result.pixel0_output = random_switch_statement(result)
    result.pixel1_output = random_switch_statement(result)
    result.pixel2_output = random_switch_statement(result)
    result.pixel3_output = random_switch_statement(result)

    return result

  def randomize_algorithm(self, alg):
    
    def shuffle_output(output):
      if len(output) == 1:
        return output

      condition_indices = map(lambda item: item[0],output[:-1])
      pixel_indices = map(lambda item: item[1],output) 

      print(condition_indices,pixel_indices)

      random.shuffle(condition_indices)
      random.shuffle(pixel_indices)

      print(condition_indices,pixel_indices)

      for i in range(len(output)):
        output[i] = (condition_indices[i] if i < len(output) - 1 else 0,pixel_indices[i])

      return output

    random_no = random.randint(0,2)

    if random_no == 0:        # method 1 - shuffle outputs
      alg.pixel0_output = shuffle_output(alg.pixel0_output) 
      alg.pixel1_output = shuffle_output(alg.pixel1_output)
      alg.pixel2_output = shuffle_output(alg.pixel2_output) 
      alg.pixel3_output = shuffle_output(alg.pixel3_output) 
    elif random_no == 1:      # method 2 - shuffle conditions
      random.shuffle(alg.conditions)    # references will be corrected by normalization
    elif random_no == 2:      # method 3 - shuffle pixels
      pixels = [self.pixel0_output,self.pixel1_output,self.pixel2_output,self.pixel3_output]
      random.shuffle(pixels)

      self.pixel0_output = pixels[0]
      self.pixel1_output = pixels[1]
      self.pixel2_output = pixels[2]
      self.pixel3_output = pixels[3]

    alg.normalize()

  def combine_algorithms(self, alg1, alg2):
   result = UpscaleAlgorithm()

   random_no = random.randint(0,1)

   if random_no == 0:        # method 1 - interlace conditions and switches
     new_conditions = [None for i in range(max(len(alg1.conditions),len(alg2.conditions)))]

     for i in range(len(new_conditions)):
       if i % 2 == 0:
         new_conditions[i] = alg1.conditions[i] if i < len(alg1.conditions) else alg2.conditions[i]
       else:
         new_conditions[i] = alg2.conditions[i] if i < len(alg2.conditions) else alg1.conditions[i]

       a1 = alg1
       a2 = alg2

       if random.randint(0,1) == 0:
         a1 = alg2
         a2 = alg1

       result.conditions = new_conditions

       result.pixel0_output = a1.pixel0_output
       result.pixel1_output = a2.pixel1_output
       result.pixel2_output = a1.pixel2_output
       result.pixel3_output = a2.pixel3_output
   elif random_no == 1:      # method 2 - concatenate conditions, interlace switches
     result.conditions = alg1.conditions + alg2.conditions
     new_remap = [i + len(alg1.conditions) for i in range(len(alg2.conditions))]
     
     for condition in alg2.conditions:
       condition.remap_references(new_remap)

     def output_condition_shift(output, offset):
       return map(lambda item: (item[0] + offset,item[1]),output[:-1]) + [output[-1]]

     result.pixel0_output = alg1.pixel0_output
     result.pixel1_output = output_condition_shift(alg2.pixel1_output,len(alg1.conditions))
     result.pixel2_output = alg1.pixel2_output
     result.pixel3_output = output_condition_shift(alg2.pixel3_output,len(alg1.conditions))


   result.normalize()
     
   return result

r = RandomGenerator(10)
a1 = r.generate_random_algorithm()
a2 = r.generate_random_algorithm()

print(a1.to_python_code())
print(a2.to_python_code())
print(r.combine_algorithms(a1,a2).to_python_code())

