from PIL import Image
import random
import math

NS = 9              # neighbour size
N_MIDDLE = NS / 2
N_MAX = NS * NS - 1 # maximum pixel number in the neighbourhood
MAX_CONDITIONS = 100

IMAGE_SIZE = (512,512)
IMAGE_SIZE2 = (IMAGE_SIZE[0] * 2,IMAGE_SIZE[1] * 2)

def print_progress_info(print_string):    # prints progress about the simulation
  print("INFO: " + print_string)

TRANSFORM_ROTATE_CW = 0    # rotate 90 degrees CW
TRANSFORM_ROTATE_CCW = 1   # rotate 90 degrees CCW
TRANSFORM_FLIP_H = 2       # flip horizontally
TRANSFORM_FLIP_V = 3       # flip vertically
TRANSFORM_SHIFT_R = 4      # shift 1 pixel to the right
TRANSFORM_SHIFT_L = 5      # shift 1 pixel to the left
TRANSFORM_SHIFT_U = 6      # shift 1 pixel up
TRANSFORM_SHIFT_D = 7      # shift 1 pixel down
TRANSFORM_NONE = 8

ALL_TRANSOFMRS = [i for i in range(9)]

def pixel_error(pixel1, pixel2):
  return abs(pixel1[0] - pixel2[0]) + abs(pixel1[1] - pixel2[1]) + abs(pixel1[2] - pixel2[2])

def pix_equal(pixel1, pixel2):
  return pixel1[0] == pixel2[0] and pixel1[1] == pixel2[1] and pixel1[2] == pixel2[2]

def pix_similar(pixel1, pixel2):
  # TODO
  pass

def pix_brightness(pixel):
  return pixel[0] + pixel[1] + pixel[2]   # not like human sight but faster

def pix_brighter(pixel1, pixel2):
  return pix_brightness(pixel1) > pix_brightness(pixel2)

def transform_pixel(pixel_index, transform_type):
  if transform_type == TRANSFORM_SHIFT_R:
    return min(pixel_index + 1,(pixel_index / NS + 1) * NS - 1)
  elif transform_type == TRANSFORM_SHIFT_L:
    return max(pixel_index - 1,(pixel_index / NS) * NS)
  elif transform_type == TRANSFORM_SHIFT_U:
    new_index = pixel_index - NS
    return new_index if new_index >= 0 else pixel_index
  elif transform_type == TRANSFORM_SHIFT_D:
    new_index = pixel_index + NS
    return new_index if new_index <= N_MAX else pixel_index
  elif transform_type == TRANSFORM_FLIP_H:
    middle = pixel_index / NS * NS + N_MIDDLE
    dx = pixel_index - middle 
    return middle - dx 
  elif transform_type == TRANSFORM_FLIP_V:
    x = pixel_index % NS
    dy = pixel_index / NS - N_MIDDLE 
    return (N_MIDDLE - dy) * NS + x
  elif transform_type == TRANSFORM_ROTATE_CW:
    x = pixel_index % NS
    y = pixel_index / NS
    return transform_pixel(x * NS + y,TRANSFORM_FLIP_H)
  elif transform_type == TRANSFORM_ROTATE_CCW:
    return transform_pixel(transform_pixel(transform_pixel(pixel_index,TRANSFORM_ROTATE_CW),TRANSFORM_ROTATE_CW),TRANSFORM_ROTATE_CW)
  else:
    return pixel_index

class PythonThing(object):               # can be represented in Python

  def to_python_code(self):
    return ""

  def self_class_name(self):
    return type(self).__name__

  def get_python_constructor(self):
    return self.self_class_name() + "()"

class NeighbourhoodCondition(PythonThing):

  def __init__(self):
    self.number_of_operands = 2
    self.operands = []    # list of either conditions (for logical conditions) or numbers (pixels or parameters, for terminal conditions) 
   
  def to_python_code(self):
    return "???"

  def remap_references(self, new_remap):
    pass

  def references_condition(self, index):
    return False

  def get_python_constructor(self): 
    result = self.self_class_name() + "("

    for operand in self.operands:
      result += operand.get_python_constructor() + ","

    return result[:-1] + ")"

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

  def apply_transform(self,transform_type):
    pass

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

  def get_python_constructor(self):
    return self.self_class_name() + "(" + str(self.operands[0]) + ")"

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

   def apply_transform(self, transform_type):
     for operand in self.operands:
       operand.apply_transform(transform_type)

class ConditionRandom(ConditionLogical):

  def __init__(self):
    super(ConditionRandom,self).__init__()
    self.number_of_operands = 0

  def to_python_code(self):
    return "(random.randint(0,1) == 1)"

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
    return "(" + self.operands[0].to_python_code() + ") != (" + self.operands[1].to_python_code() + ")"

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

class ConditionPixel(NeighbourhoodCondition):

  def __init__(self): 
    super(ConditionPixel,self).__init__()

  def apply_transform(self, transform_type):
    for i in range(len(self.operands)):
      self.operands[i] = transform_pixel(self.operands[i],transform_type)

  def get_python_constructor(self): 
    result = self.self_class_name() + "("

    for operand in self.operands:
      result += str(operand) + ","

    return result[:-1] + ")"

class ConditionPixelsAreEqual(ConditionPixel):

  def __init__(self, pixel_a, pixel_b):
    super(ConditionPixelsAreEqual,self).__init__()
    self.operands = [pixel_a, pixel_b]

  def to_python_code(self):
    return "pix_equal(p[" + str(self.operands[0]) + "],p["  + str(self.operands[1]) + "])"

class ConditionPixelIsBrighter(ConditionPixel):

  def __init__(self, pixel_a, pixel_b):
    super(ConditionPixelIsBrighter,self).__init__()
    self.operands = [pixel_a, pixel_b]

  def to_python_code(self):
    return "pix_brighter(p[" + str(self.operands[0]) + "],p["  + str(self.operands[1]) + "])"

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

class UpscaleAlgorithm(PythonThing):

  def __init__(self):

    self.conditions = []
    self.pixel0_output = []     # nonempty list of tuples in format (condition_index, pixel number), last condition is ignored
    self.pixel1_output = []
    self.pixel2_output = []
    self.pixel3_output = []

  def get_python_constructor(self):
    result = "def create_alg():\n  a = " + self.self_class_name() + "()\n"

    for condition in self.conditions:
      result += "  a.conditions.append(" + condition.get_python_constructor() + ")\n"

    result += "\n  a.pixel0_output = " + str(self.pixel0_output) + "\n"
    result += "  a.pixel1_output = " + str(self.pixel1_output) + "\n"
    result += "  a.pixel2_output = " + str(self.pixel2_output) + "\n"
    result += "  a.pixel3_output = " + str(self.pixel3_output) + "\n"

    result += "  return a\n"
    return result

  # Applies the algorithm to given PIL pixels. Optional dst image
  # pixels can be provided to store the result. Total error to
  # compare pixels is returned.

  def apply_to_pixels(self, src_pixels, compare_pixels, dst_pixels=None):
    print_progress_info("executing algorithm " + str(id(self)))

    error = 0

    exec(self.to_python_code())    # create the function

    for j in range(N_MIDDLE,IMAGE_SIZE[1] - N_MIDDLE):
      for i in range(N_MIDDLE,IMAGE_SIZE[0] - N_MIDDLE):

        neighbours = []

        for y in range(-1 * N_MIDDLE,N_MIDDLE + 1):
          for x in range(-1 * N_MIDDLE,N_MIDDLE + 1):
            neighbours.append(src_pixels[(i + x,j + y)])

        pixels = alg_function(neighbours)

        dst_x = i * 2
        dst_y = j * 2

        dst_coords = (
          (dst_x,dst_y),
          (dst_x + 1,dst_y),
          (dst_x,dst_y + 1),
          (dst_x + 1,dst_y + 1)
          )

        for k in range(4):
          error += pixel_error(compare_pixels[dst_coords[k]],pixels[k])

          if dst_pixels != None:
            dst_pixels[dst_coords[k]] = pixels[k]

    return error

  def to_python_code(self):
    result = "def alg_function(p):\n"

    i = 0

    for condition in self.conditions:
      result += "  c" + str(i) + " = " + condition.to_python_code() + "\n"
      i += 1

    r = 0

    for if_statement in (self.pixel0_output,self.pixel1_output,self.pixel2_output,self.pixel3_output):
      result += "\n"

      for i in range(len(if_statement)):

        if len(if_statement) != 1:
          if i == len(if_statement) - 1:
            result += "  else:\n"
          else:
            if i == 0:
              result += "  if c" + str(if_statement[i][0]) + ":\n"
            else:
              result += "  elif c" + str(if_statement[i][0]) + ":\n"
          
          result += "  "

        result += "  r" + str(r) + " = p[" + str(if_statement[i][1]) + "] \n"

      r += 1

    return result + "\n  return (r0,r1,r2,r3)\n"

  def delete_condition(self, index):
    print_progress_info("deleting condition " +str(index))

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
    print_progress_info("normalizing algorithm " + str(id(self)))

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

    print_progress_info("condition usage: " + str(used))

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
    print_progress_info("generating random condition")

    random_number = random.randint(0,6 if (condition_index == 0 or not generate_reference) else 7)

    if random_number in (0,1) or max_depth == 0:
      pixel1 = random.randint(0,N_MAX)
      pixel2 = pixel1

      while pixel2 == pixel1:                 # make sure to have different pixels
        pixel2 = random.randint(0,N_MAX)

      if random_number == 0:
        return ConditionPixelsAreEqual( pixel1, pixel2 )
      else:
        return ConditionPixelIsBrighter( pixel1, pixel2 )
    elif random_number in (2,3,4,5,6):

      condition1 = self.generate_random_condition(condition_index, max_depth - 1,True)

      if random_number != 5:
        condition2 = self.generate_random_condition(condition_index, max_depth - 1,True)

      if random_number == 2:
        return ConditionAnd( condition1, condition2)
      elif random_number == 3:
        return ConditionOr( condition1, condition2)
      elif random_number == 4:
        return ConditionXor( condition1, condition2)
      elif random_number == 5:
        return ConditionRandom()
      else:
        return ConditionNot( condition1 )
    else:
      return ConditionReference( random.randint(0,condition_index - 1) )

  def generate_random_algorithm(self): 
    result = UpscaleAlgorithm()
    
    print_progress_info("generating random algorithm, id = " + str(id(result)))

    def random_switch_statement(alg):
      print_progress_info("generating random pixel output switch")
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
        print_progress_info("enough conditions generated")
        break

    result.pixel0_output = random_switch_statement(result)
    result.pixel1_output = random_switch_statement(result)
    result.pixel2_output = random_switch_statement(result)
    result.pixel3_output = random_switch_statement(result)

    result.normalize()

    return result

  def randomize_algorithm(self, alg):
    print_progress_info("randomizing algorithm " + str(id(alg)))

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

    random_no = random.randint(0,4)

    if random_no == 0:        # method 1 - shuffle switches
      print_progress_info("using randomizing method 1 (shuffle output switches)")
      alg.pixel0_output = shuffle_output(alg.pixel0_output) 
      alg.pixel1_output = shuffle_output(alg.pixel1_output)
      alg.pixel2_output = shuffle_output(alg.pixel2_output) 
      alg.pixel3_output = shuffle_output(alg.pixel3_output) 
    elif random_no == 1:      # method 2 - shuffle conditions
      print_progress_info("using randomizing method 2 (shuffle conditions)")
      random.shuffle(alg.conditions)    # references will be corrected by normalization
    elif random_no == 2:      # method 3 - shuffle pixels
      print_progress_info("using randomizing method 3 (shuffle output pixels)")
      pixels = [alg.pixel0_output,alg.pixel1_output,alg.pixel2_output,alg.pixel3_output]
      random.shuffle(pixels)

      alg.pixel0_output = pixels[0]
      alg.pixel1_output = pixels[1]
      alg.pixel2_output = pixels[2]
      alg.pixel3_output = pixels[3]
    elif random_no == 3:      # method 3 - combine with new random alg.
      print_progress_info("using randomizing method 4 (combine with random)")
      alg = self.combine_algorithms(alg,self.generate_random_algorithm())
    elif random_no == 4:      # method 4 - apply transform to pixels
      print_progress_info("using randomizing method 4 (pixel transform)")
      
      for c in alg.conditions:
        c.apply_transform(random.choice(ALL_TRANSOFMRS))

    alg.normalize()

  def combine_algorithms(self, alg1, alg2):
   result = UpscaleAlgorithm()
   
   print_progress_info("combining algorithms " + str(id(alg1)) + " and " + str(id(alg2)) + ", new id = " +str(id(result)))

   random_no = random.randint(0,1)

   if random_no == 0:        # method 1 - interlace conditions and switches
     print_progress_info("using combination method 1 (interlace)")

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
     print_progress_info("using combination method 2 (append)")

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

#======================

algorithm_nn = UpscaleAlgorithm()
algorithm_nn.pixel0_output = [(0,40)]
algorithm_nn.pixel1_output = [(0,40)]
algorithm_nn.pixel2_output = [(0,40)]
algorithm_nn.pixel3_output = [(0,40)]

#-----

algorithm_eagle = UpscaleAlgorithm()
algorithm_eagle.conditions.append(ConditionAnd(ConditionPixelsAreEqual(39,30),ConditionPixelsAreEqual(30,31)))
algorithm_eagle.conditions.append(ConditionAnd(ConditionPixelsAreEqual(31,32),ConditionPixelsAreEqual(32,41)))
algorithm_eagle.conditions.append(ConditionAnd(ConditionPixelsAreEqual(41,50),ConditionPixelsAreEqual(50,49)))
algorithm_eagle.conditions.append(ConditionAnd(ConditionPixelsAreEqual(49,48),ConditionPixelsAreEqual(48,39)))
algorithm_eagle.pixel0_output = [(0,30),(0,40)]
algorithm_eagle.pixel1_output = [(1,32),(0,40)]
algorithm_eagle.pixel2_output = [(3,48),(0,40)]
algorithm_eagle.pixel3_output = [(2,50),(0,40)]

#-----

algorithm_linear = UpscaleAlgorithm()

algorithm_linear.conditions.append(ConditionRandom())
algorithm_linear.conditions.append(ConditionRandom())
algorithm_linear.conditions.append(ConditionRandom())

algorithm_linear.pixel0_output = [(0,30),(1,31),(2,40),(0,39)]
algorithm_linear.pixel1_output = [(0,31),(1,40),(2,39),(0,30)]
algorithm_linear.pixel2_output = [(0,39),(1,30),(2,31),(0,40)]
algorithm_linear.pixel3_output = [(0,40),(1,39),(2,30),(0,31)]

#-----

r = RandomGenerator(10)
a1 = r.generate_random_algorithm()

print(a1.to_python_code())

src_image = Image.open("test_training.png")  
src_pixels = src_image.load()

dst_image = Image.new("RGB", IMAGE_SIZE2, "white")
dst_pixels = dst_image.load()

cmp_image = Image.open("test_training_manual_upscale.png")
cmp_pixels = cmp_image.load()

print("error: " + str(algorithm_nn.apply_to_pixels(src_pixels,cmp_pixels,None)))
print("error: " + str(algorithm_eagle.apply_to_pixels(src_pixels,cmp_pixels,None)))
print("error: " + str(algorithm_linear.apply_to_pixels(src_pixels,cmp_pixels,dst_pixels)))

dst_image.save("result.png","PNG")

#r = RandomGenerator(308)
#a1 = r.generate_random_algorithm()
#a2 = r.generate_random_algorithm()

#print(a1.to_python_code())
#print(a1.get_python_constructor())
