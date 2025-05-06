students = {
    'Alice': 15,
    'kiran': 20,
    'kashaf': 12,
    'nimra': 18

}
print(students)
students["huda"]=24
print(students)
students['kashaf']=25
print(students)

# look for student with highest marks
x = 0
top_student = ""

for i in students:
    if students[i] > x:
        x = students[i]
        top_student = i

print(f"The student with the highest marks is {top_student} with {x} marks.")

a=7%8
print(a)
for index,student in enumerate(students):
    print(index,student)

for x in "banana":
    print(x)