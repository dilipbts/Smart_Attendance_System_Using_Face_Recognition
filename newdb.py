from app import db, User
from flask_bcrypt import generate_password_hash
from app import app

# Create an application context
with app.app_context():
    # Add a teacher
    teacher = User(
        username="Guru",
        password=generate_password_hash("1234").decode('utf-8'),
        role="Teacher"
    )

    # Add a student
    student = User(
        username="Dilip DK",
        password=generate_password_hash("demonking").decode('utf-8'),
        role="Student"
    )

    # Add users to the database and commit
    db.session.add(teacher)
    db.session.add(student)
    db.session.commit()

    print("Users added successfully!")
