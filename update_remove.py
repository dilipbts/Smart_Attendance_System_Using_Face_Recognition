from flask_bcrypt import generate_password_hash
from app import db, User
from app import app

# Create an application context
with app.app_context():
    def remove_user(username):
        """Remove a user by username."""
        user_to_remove = User.query.filter_by(username=username).first()
        if user_to_remove:
            db.session.delete(user_to_remove)
            db.session.commit()
            print(f"User '{username}' has been removed.")
        else:
            print(f"User '{username}' does not exist.")


    def update_password(username, new_password):
        """Update a user's password by username."""
        user_to_update = User.query.filter_by(username=username).first()
        if user_to_update:
            hashed_password = generate_password_hash(new_password).decode('utf-8')
            user_to_update.password = hashed_password
            db.session.commit()
            print(f"Password for user '{username}' has been updated.")
        else:
            print(f"User '{username}' does not exist.")


    # Perform actions: Remove a user and/or update a password
    username_to_remove = "teacher01"  # Specify the username to remove
    username_to_update = "student01"  # Specify the username to update
    new_password = "new_student_pass"  # Specify the new password

    # Remove user
    if username_to_remove:
        remove_user(username_to_remove)

    # Update password
    if username_to_update and new_password:
        update_password(username_to_update, new_password)
