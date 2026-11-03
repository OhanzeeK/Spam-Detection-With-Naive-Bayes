//Necessary Imports
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions._

//Importing the email data
val raw_data = spark.read.option("header", "true").option("inferSchema", "true").csv("C:/Users/0106o/OneDrive/Pictures/Documents/Classwork/Big Data Analytics/Assignment4/naive_bayes_spam.csv")

//Cleaning null rows to avoid errors
val data = raw_data.na.drop("all")
println("Original Data: ")
data.show()
println("\n")

//Converting the words in each email into individual tokens and adding an array column will each message's split words
val tokenizer = new RegexTokenizer()
.setInputCol("text")
.setOutputCol("words")
.setPattern("\\s+")
.setToLowercase(true)

val tokenized_data = tokenizer.transform(data)

//Training data is the first half of the original data
var training_data = tokenized_data.filter("email_id <= 'EMAIL00601'")

//Testing data is the second half of the original data
var testing_data = tokenized_data.filter("email_id > 'EMAIL00601'")


//Learning a vocabulary from the dataset
val vectorizer = new CountVectorizer()
.setInputCol("words")
.setOutputCol("features")
.setVocabSize(300)

val vectorizer_model = vectorizer.fit(training_data)

training_data = vectorizer_model.transform(training_data)
testing_data = vectorizer_model.transform(testing_data)


//Displaying the dataset with the tokenized words and new vocabulary
println("Data with tokens and vocabulary applied:")
training_data.show()
println("\n")

val bayes = new NaiveBayes()
.setLabelCol("label")
.setModelType("multinomial")
.setFeaturesCol("features")

val bayesModel = bayes.fit(training_data)

training_data = bayesModel.transform(training_data)
testing_data = bayesModel.transform(testing_data)

//Displaying Bayes Model results on testing data
println("Data with Bayes Model applied: ")
testing_data.select("email_id","label","rawPrediction","probability","prediction").show()
println("\n")

//Calculating the accuracy of predictions on the testing data
val evaluator = new MulticlassClassificationEvaluator()
.setLabelCol("label")
.setPredictionCol("prediction")
.setMetricName("accuracy")

var accuracy = evaluator.evaluate(testing_data)

println("Test data prediction accuracy: " + f"${accuracy * 100}%.2f%%")
println()

//Predicting whether custom emails are spam
val newEmails = Seq(
("EMAIL01201", "Free trip to barbados enter now", 1.0),
("EMAIL01202", "May I borrow your notes from class", 0.0),
("EMAIL01203", "Event winner claim your gift card now", 1.0),
("EMAIL01204", "High chance of rain this saturday afternoon", 0.0),
("EMAIL01205", "Your new xfinity bill just came in", 0.0),
)

var new_test_data = newEmails.toDF("email_id","text","label")

new_test_data = bayesModel.transform(vectorizer_model.transform(tokenizer.transform(new_test_data)))

println("New emails and predictions: ")
new_test_data.select("text","label","probability","prediction").show()
println("\n")

accuracy = evaluator.evaluate(new_test_data)


println("New email prediction accuracy: " + f"${accuracy * 100}%.2f%%")
//Note: The word "high" in email 01204 might be why it was flagged as spam