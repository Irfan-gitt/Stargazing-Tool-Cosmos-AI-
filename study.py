from pydantic import BaseModel


class User (BaseModel):
    name: str
    age: int
    address: str
    uniqueid: int


user_name = input("Enter your name: ")
user_age = input("Enter your age:  ")
user_address = input("Enter your adress: ")


class Final(User):
    User.name = user_name
    User.age = user_age
    User.address = user_address


print(User.name)
