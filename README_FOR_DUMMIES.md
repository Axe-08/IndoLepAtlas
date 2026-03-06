# SO You want to contribute to the project?

Be a little miner like me? For the Butterflies?

Well You my friend are at the right place!

Here's what you gotta do:

> **Note:** These commands will work in any Unix shell. Well, if they don't work in your corporate ass Windows just Google or AI the alternative commands. Y'all are CS students so I doubt you will need it, but anyways here are the steps.

## 1. Clone Git Repo

```bash 
git clone https://github.com/AXE8/IndoLepAtlas.git
cd IndoLepAtlas        
```

## 2. Create Virtual Environment and Activate

Windows activation might be a bit different so Google it. Also install the `requirements.txt`:

```bash 
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Set the `.env` File

Rename the `.env.example` to just `.env` and inside it replace `hf_token_here` with your own token or my token. 

You might ask, *"Mr. Genius I don't have a token (cryface)"* — go to the [Get Your HF Token](#get-your-hf-token) section for instructions.

## 4. Pull the Lever!

Run:

```bash
python3 crawler.py --chunk "I'll tell you in whatsapp" --total-chunks 5
```
    
## 5. Watch the script run

Watch the script run or leave it be, idc. 

Keep the terminal alive: don't close it, don't let your laptop sleep, don't let it die, don't let it do anything, just let it run.

---

## Get your HF Token:

1. Go to [huggingface.co](https://huggingface.co)
2. Create or login with your account
3. Tell me your username and I will make you contributor of the repo
4. Or just use my own token (here, `HF_TOKEN=hf_**********************************`) — **beware you'll see no contribution of your own if you use my token**
5. After I make you contributor, go to **Settings -> Access Tokens -> New Token (write)**, copy it and paste it in `.env` file.
   * Alternatively go here: [https://huggingface.co/settings/tokens/new?tokenType=write](https://huggingface.co/settings/tokens/new?tokenType=write)
   * Create with name of your choice and copy the token and paste it in `.env` file.