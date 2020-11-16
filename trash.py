import re

pattern = r"^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z]).{5,}(?!^MTC)$")
