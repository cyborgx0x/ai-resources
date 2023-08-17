
```
create a python dictionary with following key: 
email, password_hash, password_crack, source_data, country, username

Detail of each field:

email: the email of account
password_hash: the password_hash of account, it can be any kind of hashing
password_crack: the raw text password, it can be any random value
source_data: the source of data, it can be from various site that provide account leak
country: any country in the world

here is the example:
{
    "email": "zock1122@gmail.com",
    "password_hash": "$2a$10$c28e9F3KNimLsn/2xhanGOlYG0ECGpmVER4tXmoaalBfFHg13/7o.",
    "password_crack": None,
    "source_data": "raidforums",
    "country": "Malaysia",
    "username": "zock",
}

one of two field: email or username can be None
one of two field: password_crack and password_hash can be None
```
Result: The result seem promise. So I check another prompt to ensure that i will generate a new sample


```
create another dictionary with different information
```
Result: It good! Let create another 10 samples

```
create 10 another examples
```

Finally, add all the dictionary to a list
```

```