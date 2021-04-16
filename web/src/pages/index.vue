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

            <div class="group" v-for="(group, index) in feature_maps" :key="index">
                <div class="layer" v-for="(value, name) in group" :key="name">
                    <div class="name">{{name}}</div>
                    <img :src="createURL(value)"/>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    data() {
        return {
            feature_maps: [],
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
            this.feature_maps = data
        }
    },
    methods: {
        click() {
            this.$socket.emit("predict", { model : this.model, input : this.x });
        },
         createURL(v) {
            return window.URL.createObjectURL(new Blob([v]))
        }
    }
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
