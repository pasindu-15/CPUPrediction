from flask import Flask, jsonify, request

import sys

import svr_regression

# sys.path.append('/Users/pasindu/Documents/Msc/ResearchProject/project/MLModel/CPUPrediction/preprocessing')
# import


app = Flask(__name__)

# Sample data




# Route to get all books
@app.route('/cpu-prediction', methods=['GET'])
def get_books():
    # v = svr_regression.get_predicted_cpu([3827.0, 0.010416260256075255, 193.0, 139.0])
    v = svr_regression.get_predicted_cpu([3230.0, 0.010447052321981423, 137.0, 129.0])
    response = {"prediction": v[0][0]}

    return jsonify(response)


# Route to get a specific book by ID
# @app.route('/books/<int:id>', methods=['GET'])
# def get_book(id):
#     book = next((book for book in books if book['id'] == id), None)
#     if book:
#         return jsonify(book)
#     else:
#         return jsonify({"error": "Book not found"}), 404

# Route to add a new book
# @app.route('/books', methods=['POST'])
# def add_book():
#     new_book = request.json
#     books.append(new_book)
#     return jsonify(new_book), 201
#
# # Route to update a book
# @app.route('/books/<int:id>', methods=['PUT'])
# def update_book(id):
#     book = next((book for book in books if book['id'] == id), None)
#     if book:
#         book.update(request.json)
#         return jsonify(book)
#     else:
#         return jsonify({"error": "Book not found"}), 404
#
# # Route to delete a book
# @app.route('/books/<int:id>', methods=['DELETE'])
# def delete_book(id):
#     global books
#     books = [book for book in books if book['id'] != id]
#     return jsonify({"message": "Book deleted"}), 200

if __name__ == '__main__':
    app.run(debug=True)
