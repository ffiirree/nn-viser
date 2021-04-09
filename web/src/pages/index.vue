<template>
    <div class="page">
        <div class="menu">
            <div class="item">
                <div class="title">Model</div>
                <el-input size="small" v-model="model"/>
            </div>
            <div class="item">
                <div class="title">Input</div>
                <el-input size="small" v-model="x"/>
            </div>
            <el-button class="run-btn" size="small" type="primary" @click="click">RUN</el-button>
        </div>
        <div class="network">
            <div class="group">
                <div class="layer">
                    <img :src="input"/>
                </div>
            </div>
            
            <div class="group">
                <div class="layer"><img :src="conv1"/></div>
                <div class="layer"><img :src="relu1"/></div>
            </div>

            <div class="group">
                <div class="layer"><img :src="conv2"/></div>
                <div class="layer"><img :src="relu2"/></div>
            </div>

            <div class="group">
                <div class="layer"><img :src="conv3"/></div>
                <div class="layer"><img :src="relu3"/></div>
            </div>

            <div class="group">
                <div class="layer"><img :src="conv4"/></div>
                <div class="layer"><img :src="relu4"/></div>
            </div>

            <div class="group">
                <div class="layer"><img :src="conv5"/></div>
                <div class="layer"><img :src="relu5"/></div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    data() {
        return {
            input: null,
            conv1: null,
            relu1: null,
            conv2: null,
            relu2: null,
            conv3: null,
            relu3: null,
            conv4: null,
            relu4: null,
            conv5: null,
            relu5: null,
            model: 'logs/mnist_tiny_11.pth',
            x: 'mnist/image_0.png',
        };
    },
    created() {},
    sockets: {
        connect() {
            console.log(`socket connected.`);
        },

        disconnect() {
            console.log("socket disconnected");
        },

        message(data) {
            console.log( 'from server:' + data)
        },
        net(data) {

            this.input = window.URL.createObjectURL(new Blob([data.input]))

            this.conv1 = window.URL.createObjectURL(new Blob([data.conv1]))
            this.relu1 = window.URL.createObjectURL(new Blob([data.relu1]))

            this.conv2 = window.URL.createObjectURL(new Blob([data.conv2]))
            this.relu2 = window.URL.createObjectURL(new Blob([data.relu2]))

            this.conv3 = window.URL.createObjectURL(new Blob([data.conv3]))
            this.relu3 = window.URL.createObjectURL(new Blob([data.relu3]))

            this.conv4 = window.URL.createObjectURL(new Blob([data.conv4]))
            this.relu4 = window.URL.createObjectURL(new Blob([data.relu4]))

            this.conv5 = window.URL.createObjectURL(new Blob([data.conv5]))
            this.relu5 = window.URL.createObjectURL(new Blob([data.relu5]))
        }
    },
    methods: {
        click() {
            this.$socket.emit("predict", { model : this.model, input : this.x });
        },
    },
};
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
.page {
    height: 100%;
    background: #efefef;

    display: flex;
    flex-flow: column;

    .menu {
        flex: 0 0 auto;
        padding: 20px 30px;
        box-sizing: border-box;

        .item {
            display: flex;
            flex-flow: row;
            height: 50px;
            align-items: center;

            .title {
                width: 100px;
            }
        }

        .run-btn {
            float: right;
        }
    }
    .network {
        flex: 1 1 auto;
        background: #ffffff;
        padding: 25px;

        display: flex;
        flex: row;
        align-items: center;

        .group {
            flex: 1 1 auto;

            display: flex;
            flex-flow: row;

            .layer {
                margin-right: 30px;
                img {
                    width: 48px;
                }
            }

            .layer:last-child {
                margin-right: 0;
            }
        }
    }
}

</style>
