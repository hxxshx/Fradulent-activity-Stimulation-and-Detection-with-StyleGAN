# include necessary libraries
!pip install transformers datasets torch

#dataset path
import pandas as pd

df = pd.read_csv("/content/preprocessed_spamm.csv")
print(df.head())

#data preprocessing
import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    return text.strip()

df['cleaned_text'] = df['Message'].apply(clean_text)

#tokenization
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
df['tokenized'] = df['cleaned_text'].apply(  
      lambda x: tokenizer(    
            x, truncation=True, padding="max_length", max_length=50, return_tensors="pt"  
              )
    )

#import model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

model = AutoModelForCausalLM.from_pretrained("gpt2")

from datasets import Dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
df['tokenized'] = df['cleaned_text'].apply(
    lambda x: tokenizer(x, truncation=True, padding="max_length", max_length=50)
)
df['input_ids'] = df['tokenized'].apply(lambda x: x['input_ids'])
df['attention_mask'] = df['tokenized'].apply(lambda x: x['attention_mask'])
dataset = Dataset.from_pandas(df[['input_ids', 'attention_mask']])
print(dataset[0])


#list training arguments
training_args = TrainingArguments(
    output_dir="./phishing-gpt",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
)

#model training
from datasets import DatasetDict
train_test = dataset.train_test_split(test_size=0.2)
dataset = DatasetDict({
    "train": train_test["train"],
    "test": train_test["test"]
})
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
# generate phishing sms
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import pandas as pd
import os


model_path = "./fine_tuned_gpt_model"
tokenizer_path = "./fine_tuned_gpt_tokenizer"

if not os.path.exists(model_path):
    raise ValueError(f"Model path '{model_path}' does not exist. Please check the path.")

if not os.path.exists(tokenizer_path):
    raise ValueError(f"Tokenizer path '{tokenizer_path}' does not exist. Please check the path.")


try:
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
except Exception as e:
    raise RuntimeError(f"Error loading model/tokenizer: {e}")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prompts for phishing messages
prompts = ["Your account has been temporarily suspended due to suspicious activity, verify your details now.",
    "We detected an unauthorized login attempt on your account, secure it immediately to prevent fraud.",
    "Your payment was unsuccessful, update your billing information to avoid service interruption.",
    "Urgent security alert, your account is at risk, click the link to verify your identity.",
    "Your email storage is full, upgrade now to continue receiving important messages.",
    "Congratulations, you have won an exclusive prize, claim it before it expires.",
    "Your bank account is locked due to unusual transactions, verify now to restore access.",
    "Your subscription is expiring soon, renew today to avoid losing premium benefits.",
    "We noticed suspicious activity on your card, confirm your details to prevent unauthorized charges.",
    "Your cloud storage is at full capacity, upgrade your plan to continue saving files.",
    "Action required, verify your account to prevent deactivation due to security concerns.",
    "Your recent order could not be processed, update your payment details to complete the transaction.",
    "Your online banking access has been restricted, confirm your credentials to restore full functionality.",
    "Your password has been reset successfully, if this wasn't you, secure your account immediately.",
    "Your social media account was accessed from a new location, confirm if it was you.",
    "Your phone bill is overdue, pay now to avoid service disruption.",
    "Your email verification is pending, complete it now to continue using our services.",
    "Urgent alert, your tax refund is ready to be claimed, verify your details before it expires.",
    "Your mobile service provider detected unusual charges, confirm them to avoid account suspension.",
    "A new message from customer support is waiting for you, click the link to view it.",
    "Your antivirus subscription has expired, renew now to stay protected from online threats.",
    "Your internet provider requires an urgent update on your billing details, verify now.",
    "We have detected multiple failed login attempts, secure your account immediately.",
    "Your membership renewal is due, confirm your payment details to maintain your benefits.",
    "A package is on hold due to an incorrect address, update your shipping details now.",
    "Your electricity bill is overdue, settle it now to avoid service disconnection.",
    "Your medical insurance needs renewal, act now to ensure continuous coverage.",
    "Your recent booking has been modified, check the updated details immediately.",
    "Your online banking session has expired, log in now to avoid account suspension.",
    "A refund has been issued to your account, verify your details to receive the amount.",
    "Your social security benefits are temporarily on hold, update your information to continue receiving them.",
    "An important security patch is available for your account, install it now.",
    "Your online wallet balance is running low, top up now to continue transactions.",
    "A new voicemail from an unknown number is waiting, click to listen now.",
    "Your credit score has been updated, view the report now for the latest changes.",
    "Your subscription auto-renewal failed, update your payment details to continue uninterrupted service.",
    "A suspicious transaction was detected, verify if this purchase was made by you.",
    "Your government benefits application is incomplete, submit missing documents to avoid rejection.",
    "Your email has been flagged for suspicious activity, verify now to prevent account suspension.",
    "Your tax filing requires additional verification, update your details to avoid penalties.",
    "Your property tax bill is due soon, settle the payment to avoid extra charges.",
    "Your airline reservation has been modified, check the updated details immediately.",
    "Your lottery prize is waiting, claim it before the deadline expires.",
    "Your student loan payment is due, submit the payment to avoid late fees.",
    "Your mortgage approval is pending, verify your documents to complete the process.",
    "Your streaming subscription will expire soon, renew today for uninterrupted access.",
    "Your recent login attempt failed, reset your password if you suspect unauthorized access.",
    "A new friend request is pending, accept now to connect with them.",
    "Your app store purchase was unsuccessful, retry payment to complete the transaction.",
    "A special bonus reward is waiting for you, claim it before the offer ends.",
    "Your security settings have been changed, confirm if this was you.",
    "Your account has been flagged for multiple failed logins, secure it now.",
    "Your domain name registration is expiring, renew now to retain ownership.",
    "Your loan application requires additional verification, submit your documents now.",
    "Your credit card was charged for an unrecognized purchase, review it immediately.",
    "Your delivery is on hold due to missing address details, update now for successful delivery.",
    "Your investment account has been credited with a bonus, check your balance now.",
    "A payment dispute has been raised on your transaction, confirm if this was you.",
    "Your e-wallet needs authentication to continue transactions, verify now.",
    "Your new online banking statement is available, view your transactions now.",
    "Your employer has sent you an urgent document, access it now for review.",
    "Your health insurance claim has been processed, check the status now.",
    "Your hotel booking is incomplete, confirm details to avoid cancellation.",
    "Your cashback reward has been credited, claim it before the expiration date.",
    "Your online shopping cart is about to expire, complete the purchase now.",
    "Your subscription trial is ending soon, upgrade now to continue using premium features.",
    "Your favorite store has issued a discount voucher, redeem it before it expires.",
    "Your home security system detected unusual activity, check alerts now.",
    "A fraudulent login attempt was blocked, secure your account to prevent further risk.",
    "Your prepaid mobile balance is running low, recharge now to continue usage.",
    "Your account activity was flagged as suspicious, verify your identity now.",
    "A tax refund request has been generated under your name, confirm if this is valid.",
    "Your new job offer letter is ready, download and review the details now.",
    "Your resume was shortlisted for an interview, schedule your appointment now.",
    "Your travel insurance expires soon, renew now for uninterrupted coverage.",
    "Your banking profile needs urgent verification, failure to do so may result in restrictions.",
    "Your stored payment method has expired, update now to avoid declined transactions.",
    "Your credit card limit has been increased, check the new limit now.",
    "Your data plan is about to run out, recharge now for uninterrupted service.",
    "Your business loan application has been reviewed, update missing details to proceed.",
    "Your mobile banking account needs identity verification, complete it now to avoid restrictions.",
    "Your unpaid toll fees are due, settle them now to avoid extra charges.",
    "Your online conference link has changed, access the updated link now.",
    "Your automatic bill payment failed, resolve the issue now to prevent late fees.",
    "Your email attachments failed to send, check your settings and resend.",
    "Your workplace has issued a new security protocol, review it now.",
    "Your lost item claim has been processed, track the recovery status now.",
    "Your missed package delivery needs rescheduling, update your preferences now.",
    "Your job application status has changed, check the latest update now.",
    "Your student scholarship application requires final verification, confirm details now.",
    "Your online tax portal requires authentication, verify your identity to proceed.",
    "Your education loan repayment schedule has been revised, check new details now.",
    "Your airline loyalty points are about to expire, redeem them for rewards now."
]
# Generate phishing messages
generated_messages = []

for prompt in prompts:
    try:
        output = generator(prompt, max_length=50, truncation=True)
        generated_text = output[0]['generated_text']
        generated_messages.append({'prompt': prompt, 'generated_text': generated_text})
    except Exception as e:
        print(f"Error generating text for prompt '{prompt}': {e}")

# Convert to DataFrame
df = pd.DataFrame(generated_messages)

# Save to CSV
csv_path = "/content/generated_phishing_messages.csv"
df.to_csv(csv_path, index=False)

print(f"Phishing messages have been generated and saved to '{csv_path}'")


#save the model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.save_pretrained("./fine_tuned_gpt_model")
tokenizer.save_pretrained("./fine_tuned_gpt_tokenizer")
print("Model and tokenizer saved successfully!")


